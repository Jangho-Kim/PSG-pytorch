import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

### Quantization modules ###
class post_training_weight(nn.Module):
    def __init__(self,lambda_s,n_bits, use_fp, activation, momentum=0.1):
        super(post_training_weight, self).__init__()
        self.n_bits = n_bits
        self.lambda_s = lambda_s
        # type of quantization
        self.activation = activation
        if activation:
            self.type_q = min_max_post_training_asymmetric
        else :
            self.type_q = min_max_post_training_symmetric if use_fp else min_max_post_training_symmetric_lp
        self.x_min = None
        self.x_max = None
        self.momentum = momentum
        self.fixed_range = False
        self.quantizer = None
        # use running mean
        self.use_running_mean = True

    def update_range(self,data):
        if  self.fixed_range:
            self.quantizer = self.type_q(self.lambda_s,self.n_bits,self.x_min, self.x_max)
            return

        x_min = min(0., float(data.min()))
        x_max = max(0., float(data.max()))

        if self.use_running_mean:
            if self.x_min is not None:
                self.x_min = self.momentum * x_min + (1 - self.momentum) * (self.x_min or x_min)
                self.x_max = self.momentum * x_max + (1 - self.momentum) * (self.x_max or x_max)
            elif self.x_min is None:
                self.x_min = x_min
                self.x_max = x_max
        else:
            self.x_min = x_min
            self.x_max = x_max

        self.quantizer = self.type_q(self.lambda_s,self.n_bits, self.x_min, self.x_max)

    def change_range_mode(self,Boolean):
        self.fixed_range = Boolean

    def forward(self, x):
        self.update_range(x)
        return_value = self.quantizer.return_scale_value(x)
        return self.quantizer(x), return_value


class min_max_post_training_asymmetric(autograd.Function):
    def __init__(self,beta, n_bits, x_min, x_max):
        super(min_max_post_training_asymmetric, self).__init__()
        self.beta = beta
        if n_bits == 0:
            return None
        else:
            lower = 0
            upper = 2 ** n_bits
        # np.arange upper -1 so the range will be 0~255
        self.constraint = np.arange(lower, upper)
        self.valmin = float(self.constraint.min())
        self.valmax = float(self.constraint.max())

        self.n_levels = 2 ** (n_bits)
        self.delta = float(x_max) / (self.n_levels - 1)

    def forward(self, *args, **kwargs):
        x = args[0]
        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin , max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)
        return x_restore

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        return grad_top

    def return_scale_value(self, x):
        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)
        scale_value = torch.abs(x-x_restore)
        return scale_value

### Full precision modules (fowarding with FP weights) ###
class min_max_post_training_symmetric(autograd.Function):
    def __init__(self, beta, n_bits, x_min, x_max):
        super(min_max_post_training_symmetric, self).__init__()
        self.iter = 0
        self.beta = beta
        if n_bits == 0:
            return None
        else:
            # Restricted Mode
            lower = -2 ** (n_bits - 1) + 1
            upper = 2 ** (n_bits - 1)

        self.constraint = np.arange(lower, upper)
        self.valmin = float(self.constraint.min())
        self.valmax = float(self.constraint.max())
        x_absmax = max(abs(x_min), x_max)
        ### Full range ###
        # self.n_levels = 2 ** (n_bits) - 1
        # self.delta = float(x_absmax) / (self.n_levels / 2)
        ##################
        self.n_levels = 2 ** (n_bits-1)   # Restricted range
        self.delta = float(x_absmax) / (self.n_levels - 1)

    def forward(self, *args, **kwargs):
        x = args[0]
        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)
        scale = torch.abs(x-x_restore)
        self.save_for_backward(scale)
        return x

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        scale = self.saved_tensors[0]
        # return grad_top *scale*self.beta / scale.max() # Non-seperable Scaling
        return grad_top * scale * self.beta  # Vanilla

    def return_scale_value(self, x):
        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)
        scale_value = torch.abs(x-x_restore)
        return scale_value

### Low precision modules (fowarding with LP weights) ###
class min_max_post_training_symmetric_lp(autograd.Function):
    def __init__(self, beta, n_bits, x_min, x_max):
        super(min_max_post_training_symmetric_lp, self).__init__()
        self.iter = 0
        self.beta = beta
        if n_bits == 0:
            return None
        else:
            lower = -2 ** (n_bits - 1) + 1
            upper = 2 ** (n_bits - 1)

        self.constraint = np.arange(lower, upper)
        self.valmin = float(self.constraint.min())
        self.valmax = float(self.constraint.max())
        x_absmax = max(abs(x_min), x_max)
        x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
        self.n_levels = 2 ** (n_bits-1)
        self.delta = float(x_absmax) / (self.n_levels - 1)

    def forward(self, *args, **kwargs):
        x = args[0]

        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)

        scale = torch.abs(x-x_restore)
        self.save_for_backward(scale)
        return x_restore

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        return grad_top

    def return_scale_value(self, x):
        lambda_s = self.delta
        x_lambda_s = torch.div(x, lambda_s)
        x_clip = F.hardtanh(x_lambda_s, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        x_restore = torch.mul(x_round, lambda_s)
        scale_value = torch.abs(x-x_restore)
        return scale_value


### Network module ###
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,stride,
                                     padding, dilation, groups, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Conv2d_minmax(nn.Conv2d):
    def __init__(self,lambda_s,n_bits,in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, use_fp=True):
        super(Conv2d_minmax, self).__init__(in_channels, out_channels, kernel_size,stride,
                                     padding, dilation, groups, bias)

        self.register_buffer('scale_value', torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        self.use_fp = use_fp
        # Activation quant. is not needed for Conv. module
        self.wquantizer = post_training_weight(lambda_s,n_bits, use_fp, activation=False)

    def forward(self, x):
        weight, return_value = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        if self.use_fp:
            self.scale_value = return_value
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Linear_minmax(nn.Linear):
    def __init__(self,lambda_s,n_bits, in_features, out_features, use_fp, activation=False, bias=True):
        super(Linear_minmax, self).__init__(in_features, out_features, bias)
        self.wquantizer = post_training_weight(lambda_s,n_bits, use_fp, activation=False)
        self.bquantizer = post_training_weight(lambda_s,n_bits, use_fp, activation=False)
        self.register_buffer('wscale_value', torch.rand(out_features, in_features))
        self.register_buffer('bscale_value', torch.rand(out_features))
        self.use_bias = bias
        self.use_fp = use_fp

    def forward(self, x):
        weight, wreturn_value = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        if self.use_bias:
            bias, breturn_value = self.bias if self.bquantizer is None else self.bquantizer(self.bias)
        else :
            bias, breturn_value = None, None
        if self.use_fp:
            self.wscale_value = wreturn_value
            self.bscale_value = breturn_value
        return F.linear(x, weight, bias)

class ReLU_quant(nn.ReLU):
    def __init__(self, a_bits):
        super(ReLU_quant, self).__init__()
        self.aquantizer = post_training_weight(1, a_bits, use_fp=False, activation=True)

    def forward(self, x):
        x_quant , _ = x if self.aquantizer is None else self.aquantizer(x)
        return x_quant # ReLU function is applied inside aquantizer


