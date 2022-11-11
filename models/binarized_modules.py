import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
import global_variables as gv

def safmap_dcrxb(saf0, saf1, weight):
    fmp = torch.rand(weight.size())
    fmm = torch.rand(weight.size())
    fmp0 = fmp < saf0
    fmp1 = fmp > 1 - saf1
    fmm0 = fmm < saf0
    fmm1 = fmm > 1 - saf1

    return fmp0, fmp1, fmm0, fmm1

def safmap_one(saf0, saf1, weight):
    fm = torch.rand(weight.size())
    fm0 = fm < saf0
    fm1 = fm > 1 - saf1

    return fm0, fm1

def adaptsaf_dcrxb(weight, fmp0, fmp1, fmm0, fmm1):
    maxv=weight.max()
    minv=-1*weight.min()
    weightp=weight.clone()
    weightm=-1*weight.clone()
    weightp[weightp<0] = 0
    weightm[weightm<0] = 0
    weightp[fmp0 == 1] = 0
    weightp[fmp1 == 1] = maxv
    weightm[fmm0 == 1] = 0
    weightm[fmm1 == 1] = minv

    return weightp-weightm

def adaptsaf_one(weight, fm0, fm1):
    maxv=weight.max()
    minv=weight.min()
    weightc=weight.clone()
    weightc[fm0 == 1] = minv
    weightc[fm1 == 1] = maxv

    return weightc
    
def adaptvar(weight, bit, cbsize, sigma, alpha=0, LRS=1e3, HRS=1e6):
    Scale = 1. / LRS - 1. / HRS
    # print(LRS, HRS, Scale)
    # print('-----------Add variability-----------')
    nlevels = 2 ** (bit - 1)
    R_out = torch.zeros(weight.size()).type(gv.args.type)
    prog_var = torch.randn(weight.size()).type(gv.args.type)
    local_var = torch.randn(weight.size()).type(gv.args.type)
    for i in range(0, nlevels + 1):
        k = i / nlevels
        c = k * Scale + 1. / HRS  # map weight to conductance
        r = 1. / c
        local_shift = r * sigma / 5.
        lnr = math.log(r)  # typo in paper
        lnR_var = lnr + sigma * prog_var
        R_bar = torch.exp(lnR_var) + local_shift * local_var # add programming & local variability
        # print(C_bar)
        R_bar[weight != i] = 0
        # print(C_bar)
        R_out += R_bar

    # add global variability
    if alpha > 0:
        global_shift = alpha * sigma * LRS  # different constant values across different crossbars
        R = R_out.reshape(R_out.size(0),-1)
        mi = int(math.ceil(R.size(0)/float(cbsize)))
        mj = int(math.ceil(R.size(1)/float(cbsize)))
        
        R_pad = R.new().resize_(cbsize*mi, cbsize*mj)
        R_pad[:] = 1
        R_pad[:R.size(0),:R.size(1)] = R

        for i in range(mi):
            for j in range(mj):
                R_pad[cbsize*i:cbsize*(i+1), cbsize*j:cbsize*(j+1)] += global_shift * torch.randn(1).type(gv.args.type)
    
        R_out = R_pad[:R.size(0), :R.size(1)].view(R_out.size()).type(gv.args.type)
    C_out = 1. / R_out
    W_out = (C_out - 1 / HRS) / Scale
    # print('-----------Finished-----------')
    return W_out

def var_adapt_one(weight, bit, cbsize, sigma, alpha=0, LRS=1e3, HRS=1e6):
    vmin = torch.abs(torch.min(weight.clone()))
    Wm = weight.clone()+vmin
    vmax = torch.max(torch.abs(Wm.clone()))
    Wp = torch.div(Wm.clone(), vmax)
    Wp_bar = adaptvar(Wp, bit, cbsize, sigma, alpha, LRS, HRS)
    W_bar = torch.mul((Wp_bar), vmax)
    W_bar = W_bar-vmin

    return W_bar

def var_adapt_dcrxb(weight, bit, cbsize, sigma, alpha=0, LRS=1e3, HRS=1e6):
    vmax = torch.max(torch.abs(weight.clone()))
    Wp = torch.div(weight.clone(), vmax)
    Wn = torch.div(-1*weight.clone(), vmax)
    Wp[Wp < 0] = 0
    Wn[Wn < 0] = 0
    Wp_bar = adaptvar(Wp, bit, cbsize, sigma, alpha, LRS, HRS)
    Wn_bar = adaptvar(Wn, bit, cbsize, sigma, alpha, LRS, HRS)
    W_bar = torch.mul((Wp_bar - Wn_bar), vmax)

    return W_bar


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Quantize(tensor, numBits=8, quant_mode='det',  params=None):
    #tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    tensor.clamp_(-1,1)
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
    return tensor

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.out_scale = 1.
        self.saf_flag = False

    def forward(self, input):
        #print(input.shape)

        ### Quantization ###
        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not gv.args.subw:
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            #self.weight.data=Binarize(self.weight.org)
            self.weight.data=Quantize(self.weight.org, gv.args.nb)

        if gv.args.ideal:
            out = nn.functional.linear(input, self.weight) #calculate output & gradient

        else:
            with torch.no_grad():
                We = self.weight.data.clone()
                ### Variability ###
                if gv.args.sigma > 0:
                        if gv.args.dcrxb:
                            We = var_adapt_dcrxb(We, gv.args.nb, gv.args.cbsize, gv.args.sigma, gv.args.alpha)
                        else:
                            We = var_adapt_one(We, gv.args.nb, gv.args.cbsize, gv.args.sigma, gv.args.alpha)

                ### SAF ###
                if gv.args.sa0 > 0 or gv.args.sa1 > 0:
                        if gv.args.dcrxb:
                            if self.saf_flag == False:
                                fmp0, fmp1, fmm0, fmm1 = safmap_dcrxb(gv.args.sa0, gv.args.sa1, We)
                                self.weight.fmp0 = fmp0
                                self.weight.fmp1 = fmp1
                                self.weight.fmm0 = fmm0
                                self.weight.fmm1 = fmm1
                                self.saf_flag == True
                            We = adaptsaf_dcrxb(We, self.weight.fmp0, self.weight.fmp1, self.weight.fmm0, self.weight.fmm1)
                        else:
                            if self.saf_flag == False:
                                fm0, fm1 = safmap_one(gv.args.saf0, gv.args.saf1, We)
                                self.weight.fm0 = fm0
                                self.weight.fm1 = fm1
                                self.saf_flag == True
                            We = adaptsaf_one(We, self.weight.fm0, self.weight.fm1)

                ### SCN ###
                if gv.args.scn:
                    SCN = gv.net
                    cbsize = gv.args.cbsize
                    W = We.reshape(We.size(0),-1)
                    mi = int(math.ceil(W.size(0)/float(cbsize)))
                    mj = int(math.ceil(W.size(1)/float(cbsize)))

                    scn_input = W.new().resize_(mi*mj, cbsize, cbsize)
                    W_pad = W.new().resize_(cbsize*mi, cbsize*mj)
                    W_pad[:] = 1
                    W_pad[:W.size(0),:W.size(1)] = W

                    for i in range(mi):
                        for j in range(mj):
                            scn_input[i*mj+j] = W_pad[cbsize*i:cbsize*(i+1), cbsize*j:cbsize*(j+1)]

                    scn_output = SCN(scn_input)

                    for i in range(mi):
                        for j in range(mj):
                            W_pad[cbsize*i:cbsize*(i+1), cbsize*j:cbsize*(j+1)] = scn_output[i*mj+j]

                    We = W_pad[:W.size(0), :W.size(1)].view(We.size()).type(gv.args.type)

                ### Read noise ###
                if gv.args.read_noise > 0:
                    if gv.args.dcrxb:
                        Wp=We.clone()
                        Wn=-1*We.clone()
                        Wp[Wp<0] = 0
                        Wn[Wn<0] = 0
                        Wp += torch.randn(Wp.size()).type(gv.args.type)*gv.args.read_noise*Wp
                        Wn += torch.randn(Wn.size()).type(gv.args.type)*gv.args.read_noise*Wn
                        We = Wp - Wn
                    else:
                        vmin = torch.abs(torch.min(We.clone()))
                        Wm = We.clone()+vmin
                        Wm += torch.randn(Wm.size()).type(gv.args.type)*gv.args.read_noise*Wm
                        We = Wm-vmin

            out = nn.functional.linear(input, We)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.out_scale = 1.
        self.saf_flag = False


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not gv.args.subw:
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            #self.weight.data=Binarize(self.weight.org)
            self.weight.data=Quantize(self.weight.org, gv.args.nb)

        if gv.args.ideal:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        else:
            with torch.no_grad():
                We = self.weight.data.clone()
                ### Variability ###
                if gv.args.sigma > 0:
                        if gv.args.dcrxb:
                            We = var_adapt_dcrxb(We, gv.args.nb, gv.args.cbsize, gv.args.sigma, gv.args.alpha)
                        else:
                            We = var_adapt_one(We, gv.args.nb, gv.args.cbsize, gv.args.sigma, gv.args.alpha)

                ### SAF ###
                if gv.args.sa0 > 0 or gv.args.sa1 > 0:
                    with torch.no_grad():
                        if gv.args.dcrxb:
                            if self.saf_flag == False:
                                fmp0, fmp1, fmm0, fmm1 = safmap_dcrxb(gv.args.sa0, gv.args.sa1, We)
                                self.weight.fmp0 = fmp0
                                self.weight.fmp1 = fmp1
                                self.weight.fmm0 = fmm0
                                self.weight.fmm1 = fmm1
                                self.saf_flag == True
                            We = adaptsaf_dcrxb(We, self.weight.fmp0, self.weight.fmp1, self.weight.fmm0, self.weight.fmm1)
                        else:
                            if self.saf_flag == False:
                                fm0, fm1 = safmap_one(gv.args.saf0, gv.args.saf1, We)
                                self.weight.fm0 = fm0
                                self.weight.fm1 = fm1
                                self.saf_flag == True
                            We = adaptsaf_one(We, self.weight.fm0, self.weight.fm1)

                ### SCN ###
                if gv.args.scn:
                    SCN = gv.net
                    cbsize = gv.args.cbsize
                    W = We.reshape(We.size(0),-1)
                    mi = int(math.ceil(W.size(0)/float(cbsize)))
                    mj = int(math.ceil(W.size(1)/float(cbsize)))

                    scn_input = W.new().resize_(mi*mj, cbsize, cbsize)
                    W_pad = W.new().resize_(cbsize*mi, cbsize*mj)
                    W_pad[:] = 1
                    W_pad[:W.size(0),:W.size(1)] = W

                    for i in range(mi):
                        for j in range(mj):
                            scn_input[i*mj+j] = W_pad[cbsize*i:cbsize*(i+1), cbsize*j:cbsize*(j+1)]

                    scn_output = SCN(scn_input)

                    for i in range(mi):
                        for j in range(mj):
                            W_pad[cbsize*i:cbsize*(i+1), cbsize*j:cbsize*(j+1)] = scn_output[i*mj+j]

                    We = W_pad[:W.size(0), :W.size(1)].view(We.size()).type(gv.args.type)

                ### Read noise ###
                if gv.args.read_noise > 0:
                    if gv.args.dcrxb:
                        Wp=We.clone()
                        Wn=-1*We.clone()
                        Wp[Wp<0] = 0
                        Wn[Wn<0] = 0
                        Wp += torch.randn(Wp.size()).type(gv.args.type)*gv.args.read_noise*Wp
                        Wn += torch.randn(Wn.size()).type(gv.args.type)*gv.args.read_noise*Wn
                        We = Wp - Wn
                    else:
                        vmin = torch.abs(torch.min(We.clone()))
                        Wm = We.clone()+vmin
                        Wm += torch.randn(Wm.size()).type(gv.args.type)*gv.args.read_noise*Wm
                        We = Wm-vmin

            out = nn.functional.conv2d(input, We, None, self.stride,
                                                self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
