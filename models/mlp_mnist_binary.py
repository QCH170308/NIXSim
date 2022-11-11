import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
#BinarizeLinear = nn.Linear  # for fp model
#BinarizeConv2d = nn.Conv2d  # for fp model



class MLP_mnist(nn.Module):

    def __init__(self, num_classes=10):
        super(MLP_mnist, self).__init__()
        self.classifier = nn.Sequential(
            BinarizeLinear(784, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(2048, 2048),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(inplace=True),
            BinarizeLinear(2048, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax()
        )

        self.regime = {
           0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 1e-2},
           50: {'lr': 5e-3},
           100: {'lr': 1e-3},
           150: {'lr': 5e-4},
           200: {'lr': 1e-4},
        }

        #self.regime = {
        #   0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 1e-2},
        #   40: {'lr': 1e-3},
        #   80: {'lr': 1e-4},
        #   120: {'lr': 1e-5},
        #   140: {'lr': 1e-6},
        #   160: {'lr': 1e-7},
        #   180: {'lr': 1e-8},
        #   200: {'lr': 1e-9}
        #}

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return x


def mlp_mnist_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return MLP_mnist(num_classes)
