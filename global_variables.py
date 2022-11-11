import math
import torch

args = None
net = None

def bias_scale(x):
    a,b,c,d = 24.99e-6,2.026,-24.99e-6,-1.893 #LRS coefs
    return a*math.exp(b*x)+c*math.exp(d*x)

def f(x,a,b,c,d):
    return x.mul(b).exp().mul(a).add(x.mul(d).exp().mul(c))

def df(x,a,b,c,d):
    return x.mul(b).exp().mul(a*b).add(x.mul(d).exp().mul(c*d))

def f_num(x, state):
    a,b,c,d,aa,bb,cc = coefs[state]
    return a*math.exp(b*x)+c*math.exp(d*x)

coefs = {}
coefs['HRS'] = (3.507e-6,2.148,-3.507e-6,-2.265,2e-5,1e-5,0) #a,b,c,d,approx_a,_b,_c
coefs['LRS'] = (24.99e-6,2.026,-24.99e-6,-1.893,1e-4,2e-5,0)

def inverse_tensor(y, state):
    a,b,c,d,aa,bb,cc = coefs[state]
    #cnt = 0
    x = init_val(y,aa,bb,cc)
    err = x.clone()
    err[:] = 1
    while err.abs().max() > 1e-6:
        err = f(x,a,b,c,d)-y
        x = x - err.div(df(x,a,b,c,d))
        #cnt += 1
    return x

def init_val(y,a,b,c):
    x = torch.zeros(y.shape).type(y.type())
    yp = y[y.gt(0)]
    yn = y[y.lt(0)]
    xp = init_val_helper(yp,a,b,c)
    xn = init_val_helper(yn,-a,-b,-c)
    x[y.gt(0)]=xp
    x[y.lt(0)]=xn
    return x

def init_val_helper(y,a,b,c):
    #inverse of quadratic approximation
    return y.sub(c).mul(4*a).add(b**2).sqrt().sub(b).div(2*a)
