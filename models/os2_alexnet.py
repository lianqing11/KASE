import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['os2_alexNet', 'os2_alexnet']


model_urls = {
    'os2_alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class os2_alexNet(nn.Module):

    def __init__(self, num_classes=1000, avgpool_size=5):
        super(os2_alexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )
        self.fc0 = self.classifier
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(10, 48),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(48),
            nn.Linear(48, 24),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(24),
            nn.Linear(24, 1),
        )
        self.temp_classifier = nn.ModuleList([self.fc0, self.fc1, self.fc2])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc0(x)
        x = self.fc1(x)
        x1 = nn.functional.softmax(x)
        x1 = self.fc2(x1.detach())
        return x, x1


def os2_alexnet(pretrained=False, **kwargs):
    r"""os2_alexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = os2_alexNet(**kwargs)
    if pretrained:
       model_dict = model.state_dict()
       pretrained_dict = model_zoo.load_url(model_urls['os2_alexnet'])
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       model_dict.update(pretrained_dict)
       model.load_state_dict(model_dict)
    model.classifier = model.temp_classifier
    return model
