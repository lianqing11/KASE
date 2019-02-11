import torch.nn as nn
import math
import torch

__all__ = ['domain_classifier']

class DomainClassifier(nn.Module):
    def __init__(self, num_input):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
                                nn.Linear(512, 128), 
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 1))
    def forward(self, x):
        return self.fc(x)

def domain_classifier(num_input):
    return DomainClassifier(num_input)


