import torch
import torch.nn as nn
import pickle

class SCN(nn.Module):
    def __init__(self, cbsize):
        super(SCN, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
                )
        self.cbsize = cbsize
        self.c1 = torch.zeros((2,self.cbsize,self.cbsize)) #CMul1 param
        self.c2 = torch.zeros((1,self.cbsize,self.cbsize)) #CMul2 param

    def forward(self, x):
        y = x.mul(self.c1)
        y = y.view(-1, 1, self.cbsize, self.cbsize)
        y = self.net(y)
        y = y.view(-1, self.cbsize, self.cbsize)
        y = y.mul(self.c2)
        return y

    def type(self,t):
        super(SCN, self).type(t)
        self.c1 = self.c1.type(t)
        self.c2 = self.c2.type(t)
        return self


def scn_load(fname):
    with open(fname, 'rb') as f:
        model = pickle.load(f)
    return model
