import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch


__all__ =['resnet34','resnet50','resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class DomainAdaptModule(nn.Module):
    def _init_bn_layers(self, layers):
        self._bn_layers = layers
        self._bn_src_values = []
        self._bn_tgt_values = []
    def bn_save_source(self):
        self._bn_src_values = []
        for layer in self._bn_layers:
            self._bn_src_values.append(layer.running_mean.clone())
            self._bn_src_values.append(layer.running_var.clone())

    def bn_restore_source(self):
        for i, layer in enumerate(self._bn_layers):
            layer.running_mean.copy_(self.bn_src_values[i*2 + 0])
            layer.running_var.copy_(self._bn_src_values[i*2 + 1])

    def bn_save_target(self):
        self._bn_tgt_values = []
        for layer in self._bn_layers:
            self._bn_tgt_values.append(layer.running_mean.clone())
            self._bn_tgt_values.append(layer.running_var.clone())

    def bn_restore_target(self):
        for i, layer in enumerate(self._bn_layers):
            layer.running_mean.copy_(self._bn_tgt_values[i*2 + 0])
            layer.running_var.copy_(self._bn_tgt_values[i*2 + 1])
    def usource_layer(self, source_bn):
        for i, (src_layer, tgt_layer) in enumerate(zip(source_bn, self._bn_layers)):
            tgt_layer.running_mean = src_layer.running_mean
            tgt_layer.running_var = src_layer.running_var




class ResNet(DomainAdaptModule):

    def __init__(self, block, layers, num_classes=1000, avgpool_size=5):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc1 = nn.Sequential(
                                 nn.Linear(512*4, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(512, num_classes))
        self.fc2 = nn.Sequential(
                                 nn.Linear(num_classes, 48),
                                 nn.BatchNorm1d(48),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(48,24),
                                 nn.BatchNorm1d(24),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(24,1))

        self.classifier = nn.ModuleList([self.fc1, self.fc2])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        return x1, x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
       model_dict = model.state_dict()
       pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       model_dict.update(pretrained_dict)
       model.load_state_dict(model_dict)
       print("pretrained resnet 50...")
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
       model_dict = model.state_dict()
       pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       model_dict.update(pretrained_dict)
       model.load_state_dict(model_dict)
       print("pretrained resnet 101...")
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
       model_dict = model.state_dict()
       pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
       pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
       model_dict.update(pretrained_dict)
       model.load_state_dict(model_dict)
       print("pretrained resnet 152...")

    return model
