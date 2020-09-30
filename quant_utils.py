import torch
import torch.nn as nn
from quant_module import *


def save_bn_param(module, bn_module, n):
    quantizer = module.aquantizer
    quantizer.change_range_mode(True)
    shift = bn_module.bias
    scale = bn_module.weight
    x_min = 0
    x_max = shift + n * scale
    quantizer.x_min = x_min
    quantizer.x_max = x_max.max()

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

def search_bn(model, n):
    my_list = []
    prev = prev_name = None
    for name, m in model.named_children():
        if is_bn(prev) and is_ReLU(m):
            save_bn_param(m, prev, n)
            my_list.append((m, prev))
        search_bn(m, n)
        prev = m
        prev_name = name
    return my_list

def is_ReLU(m):
    return isinstance(m, ReLU_quant)

def compute_mse(model):
    # Computes Mean Absolute Percentage Error (MAPE) & Sum of Squares Error (SSE) of each layer
    sse_dict = {}
    mape_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, Conv2d_minmax) or isinstance(m, Linear_minmax):
                lp_weight = m.wquantizer(m.weight)[0].detach()
                fp_weight = m.weight
                mae = torch.nn.functional.l1_loss(fp_weight, lp_weight, reduction='none')
                sse = torch.nn.functional.mse_loss(fp_weight, lp_weight, reduction='sum')
                mape = mae / (torch.abs(fp_weight) + 1e-12)
                mape_dict[n] = torch.mean(mape).item()
                sse_dict[n] = sse.item()
    return mape_dict, sse_dict

