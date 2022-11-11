import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d



class MNIST(nn.Module):

    def __init__(self, num_classes=1000):
        super(MNIST, self).__init__()
        self.classifier = nn.Sequential(
            BinarizeLinear(784, 256, bias=True),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(256, 128, bias=True),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(128, 10, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return x


def mnist(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return MNIST(num_classes)
