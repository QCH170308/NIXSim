import argparse
import torch
import torch.nn as nn
import math
from scn import SCN, scn_load
import numpy as np
from statistics import mean
import logging
import timeit
from utils import *

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--cbsize', default=32, type=int,
                    help='crossbar size (default: 0, means no crossbar sim)')
parser.add_argument('--rw', default=0.1, type=float,
                    help='wire resistance')
parser.add_argument('--sigma', default=0, type=float, help='deviation value for variability')
parser.add_argument('--alpha', default=1, type=float, help='scale factor for global variability')
parser.add_argument('--sa0', default=0, type=float, help='SA0 rate')
parser.add_argument('--sa1', default=0, type=float, help='SA1 rate')
parser.add_argument('--read_noise', default=0, type=float, help='read noise scaler')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--input_quant', default=1, type=int, help='Input Quantization in Binary Linear and Conv')
parser.add_argument('--weight_quant', default=1, type=int, help='Weight Quantization in Binary Linear and Conv')
parser.add_argument('--scn', default='', type=str, help='scn model')
parser.add_argument('--dcrxb', default='true', type=str, help='use this option when you use two crossbar for experiment.')

args = parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def linquant(x, bits=4):
    cstd = 2**(bits-1)  # exclude sign bit
    cmin = -cstd
    cmax = cstd
    return torch.div(torch.clamp(torch.round(torch.mul(x, cstd)), cmin, cmax), cstd)

args.dcrxb=str2bool(args.dcrxb)

log_path = './sim_results/SCN_non/log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

if "Double" in args.scn:
    setup_logging(os.path.join(log_path,
                               '{cbsize}-{rw}-{nb}b-Double.txt'.format(cbsize=args.cbsize, rw=args.rw,
                                                                       nb=args.weight_quant)))
    logging.info("Balanced simulation")
else:
    setup_logging(os.path.join(log_path,
                               '{cbsize}-{rw}-{nb}b-Ref.txt'.format(cbsize=args.cbsize, rw=args.rw,
                                                                       nb=args.weight_quant)))
    logging.info("Unbalanced simulation")


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
    R_out = torch.zeros(weight.size()).type(args.type)
    prog_var = torch.randn(weight.size()).type(args.type)
    local_var = torch.randn(weight.size()).type(args.type)
    for i in range(0, nlevels + 1):
        k = i / nlevels
        c = k * Scale + 1. / HRS  # map weight to conductance
        r = 1. / c
        local_shift = r * sigma / 5.
        lnr = math.log(r)  # typo in paper
        lnR_var = lnr + sigma * prog_var
        R_bar = torch.exp(lnR_var) + local_shift * local_var  # add programming & local variability
        # print(C_bar)
        R_bar[weight != i] = 0
        # print(C_bar)
        R_out += R_bar

    # add global variability
    if alpha > 0:
        global_shift = alpha * sigma * LRS  # different constant values across different crossbars
        R = R_out.reshape(R_out.size(0), -1)
        mi = int(math.ceil(R.size(0) / float(cbsize)))
        mj = int(math.ceil(R.size(1) / float(cbsize)))

        R_pad = R.new().resize_(cbsize * mi, cbsize * mj)
        R_pad[:] = 1
        R_pad[:R.size(0), :R.size(1)] = R

        for i in range(mi):
            for j in range(mj):
                R_pad[cbsize * i:cbsize * (i + 1), cbsize * j:cbsize * (j + 1)] += global_shift * torch.randn(1).type(
                    args.type)

        R_out = R_pad[:R.size(0), :R.size(1)].view(R_out.size()).type(args.type)
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
    Wn = torch.div(-1 * weight.clone(), vmax)
    Wp[Wp < 0] = 0
    Wn[Wn < 0] = 0
    Wp_bar = adaptvar(Wp, bit, cbsize, sigma, alpha, LRS, HRS)
    Wn_bar = adaptvar(Wn, bit, cbsize, sigma, alpha, LRS, HRS)
    W_bar = torch.mul((Wp_bar - Wn_bar), vmax)

    return W_bar

time_list = []

for i in range(10):

    input = torch.randn([1, args.cbsize]).type(args.type)
    weight = torch.randn([args.cbsize, args.cbsize]).type(args.type)
    if args.input_quant == 1:
        input = Binarize(input)
    else:
        input = linquant(input, bits=args.input_quant)
    if args.weight_quant == 1:
        wq = Binarize(weight)
    else:
        wq = linquant(weight, bits=args.weight_quant)
    # print(input.shape)
    # print(wq.shape)
    
    ideal_out = nn.functional.linear(input, wq)

    start = timeit.default_timer()
    scn_model=scn_load(args.scn).type(args.type)
    with torch.no_grad():
        if args.sigma > 0:
            if args.dcrxb:
                wq = var_adapt_dcrxb(wq, args.weight_quant, args.cbsize, args.sigma, args.alpha)
            else:
                wq = var_adapt_one(wq, args.weight_quant, args.cbsize, args.sigma, args.alpha)
        
        #print(wq)
        if args.sa0 > 0 or args.sa1 > 0:
            if args.dcrxb:
                fmp0, fmp1, fmm0, fmm1 = safmap_dcrxb(args.sa0, args.sa1, wq)
                wq = adaptsaf_dcrxb(wq, fmp0, fmp1, fmm0, fmm1)
            else:
                fm0, fm1 = safmap_one(args.saf0, args.saf1, wq)
                wq = adaptsaf_one(wq, fm0, fm1)

        #print(wq)
        wq = scn_model(wq).reshape(wq.size())

        if args.read_noise > 0:
            if args.dcrxb:
                Wp = wq.clone()
                Wn = -1 * wq.clone()
                Wp[Wp < 0] = 0
                Wn[Wn < 0] = 0
                Wp += torch.randn(Wp.size()).type(args.type) * args.read_noise * Wp
                Wn += torch.randn(Wn.size()).type(args.type) * args.read_noise * Wn
                wq = Wp - Wn
            else:
                vmin = torch.abs(torch.min(wq.clone()))
                Wm = wq.clone() + vmin
                Wm += torch.randn(Wm.size()).type(args.type) * args.read_noise * Wm
                wq = Wm - vmin
    #print(wq.shape)
    out = nn.functional.linear(input, wq)
    end = timeit.default_timer()

    logging.info("Ideal Output:", ideal_out)
    logging.info("Expected Nonideal Output:", out)
    
    sim_time = end - start
    time_list.append(sim_time)
    total_time = sum(time_list)
    avg_time = mean(time_list)

    logging.info("\n Iteration: {0}\t"
                 "sim_time {sim_time:.8f}\t"
                 "avg_sim_time {avg_time:.8f}\t"
                 "total_sim_time {total_time:.8f}\n"
                 .format(i, sim_time=sim_time, avg_time=avg_time, total_time=total_time))