#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pickle

npydir = sys.argv[1]
fname = sys.argv[2]
cbsize = int(sys.argv[3])

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
        self.c1 = torch.zeros((self.cbsize,self.cbsize)) #CMul1 param
        self.c2 = torch.zeros((self.cbsize,self.cbsize)) #CMul2 param

    def forward(self, x):
        y = x.mul(self.c1)
        y = y.view(-1, 1, self.cbsize, self.cbsize)
        y = self.net(y)
        y = y.view(-1, self.cbsize, self.cbsize)
        y = y.mul(self.c2)
        return y

CMul = []
Wght = []
Bias = []

CMul.append(torch.from_numpy(np.load(os.path.join(npydir,'CMul1.npy'))).float())
CMul.append(torch.from_numpy(np.load(os.path.join(npydir,'CMul2.npy'))).float())
for i in range(7):
    Wght.append(torch.from_numpy(np.load(os.path.join(npydir,'Wght'+str(i+1)+'.npy'))).float())
    Bias.append(torch.from_numpy(np.load(os.path.join(npydir,'Bias'+str(i+1)+'.npy'))).float())

scn = SCN(cbsize)
scn.c1[:] = CMul[0]
scn.c2[:] = CMul[1]
with torch.no_grad():
  for i in range(7):
    scn.net[2*i].weight[:] = Wght[i]
    scn.net[2*i].bias[:] = Bias[i]

#with open('scn.model', 'wb') as f:
with open(fname, 'wb') as f:
    pickle.dump(scn, f)

print('Done!')
