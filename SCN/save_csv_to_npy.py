#!/usr/bin/env python

import numpy as np
import sys
import os

csvdir = sys.argv[1]
npydir = sys.argv[2]
files = os.listdir(csvdir)

for i in range(len(files)):
    f = files[i]
    target = os.path.join(npydir, f.split('.csv')[-2])
    param = np.loadtxt(os.path.join(csvdir, f), delimiter=',', ndmin=2)
    if f[:4]=='CMul': #CMul
        np.save(target, param)
    elif f[:4]=='Wght': #Conv weights
        np.save(target, param.reshape(param.shape[0],-1,3,3))
    elif f[:4]=='Bias': #Conv biases
        np.save(target, param.reshape(-1))
    else:
        print('Unexpected filename: '+f)

print('Done!')
