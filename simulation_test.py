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
parser.add_argument('--scn', default='', type=str, help='scn model')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--input_quant', default=1, type=int, help='Input Quantization in Binary Linear and Conv')
parser.add_argument('--weight_quant', default=1, type=int, help='Weight Quantization in Binary Linear and Conv')

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

log_path = './sim_results/SCN/log/'
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

    #print(input)
    #print(wq)
    ideal_out = nn.functional.linear(input, wq)

    start = timeit.default_timer()
    scn_model=scn_load(args.scn).type(args.type)
    with torch.no_grad():
        sim_weight = scn_model(wq).reshape(wq.size())
    #print(sim_weight.shape)
    scn_out = nn.functional.linear(input, sim_weight)
    end = timeit.default_timer()

    sim_time = end - start
    time_list.append(sim_time)
    total_time = sum(time_list)
    avg_time = mean(time_list)

    logging.info("Ideal Output:", ideal_out)
    logging.info("Expected Nonideal Output:", scn_out)

    logging.info("\n Iteration: {0}\t"
                 "sim_time {sim_time:.8f}\t"
                 "avg_sim_time {avg_time:.8f}\t"
                 "total_sim_time {total_time:.8f}\n"
                 .format(i, sim_time=sim_time, avg_time=avg_time, total_time=total_time))