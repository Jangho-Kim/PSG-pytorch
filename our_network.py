import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_module import *


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, lambda_s, w_bits, a_bits, in_planes, planes, stride=1, use_fp=True, activation_quant=True):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_minmax(lambda_s, w_bits, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, use_fp=use_fp)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = ReLU_quant(a_bits) if activation_quant else nn.ReLU()
        self.conv2 = Conv2d_minmax(lambda_s, w_bits, planes, planes, kernel_size=3, stride=1, padding=1, bias=False, use_fp=use_fp)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = ReLU_quant(a_bits) if activation_quant else nn.ReLU()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                Conv2d_minmax(lambda_s, w_bits, in_planes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=False, use_fp=use_fp),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, lambda_s=1, w_bits=4, a_bits=4, use_fp=True, activation_quant=True, quant_first_last=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.lambda_s= lambda_s
        self.act_quant = activation_quant

        if quant_first_last: # Fixed to 8 bit for 3 bits or lower
            fl_bits = 8 if w_bits < 4 else w_bits
            self.conv1 = Conv2d_minmax(lambda_s, fl_bits, 3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False, use_fp=use_fp)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.act = ReLU_quant(a_bits) if activation_quant else nn.ReLU()

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, use_fp=use_fp)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, use_fp=use_fp)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, use_fp=use_fp)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)

        if quant_first_last:
            self.fc = Linear_minmax(lambda_s, fl_bits, 64 * block.expansion, num_classes, use_fp=use_fp)
        else:
            self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_fp):
        layers = []
        layers.append(block(self.lambda_s, self.w_bits, self.a_bits, self.in_planes, planes, stride, use_fp, self.act_quant))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.lambda_s, self.w_bits, self.a_bits, self.in_planes, planes, 1, use_fp, self.act_quant))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet32(w_bits, a_bits, lambda_s, use_fp=True, activation_quant=False, quant_first_last=False):
    return ResNet(BasicBlock, [5,5,5], w_bits=w_bits, a_bits=a_bits, lambda_s=lambda_s, use_fp=use_fp, activation_quant=activation_quant, quant_first_last=quant_first_last)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, q_cfg):
        super(VGG, self).__init__()
        self.features = features
        self.q_cfg = q_cfg
        self.quant_first_last = q_cfg['quant_fl']
        self.use_fp = q_cfg['use_fp']
        fl_bit = q_cfg['w_bits'] if q_cfg['w_bits'] > 3 else 8
        fl_lambda_s = q_cfg['lambda_s']

        if self.quant_first_last:
            self.last_layer = Linear_minmax(fl_lambda_s, fl_bit, 4096, 100, use_fp=self.use_fp)
        else:
            self.last_layer = nn.Linear(4096, 100)

        self.act = ReLU_quant(self.q_cfg['a_bits']) if self.q_cfg['activation_quant'] else nn.ReLU()

        self.classifier = nn.Sequential(
            Linear_minmax(self.q_cfg['lambda_s'], self.q_cfg['w_bits'], 512, 4096, use_fp=self.use_fp, bias=True),
            self.act, # ReLUQuant
            nn.Dropout(),
            Linear_minmax(self.q_cfg['lambda_s'], self.q_cfg['w_bits'], 4096, 4096, use_fp=self.use_fp, bias=True),
            self.act, #ReLUQuant x
            nn.Dropout(),
            self.last_layer,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, q_cfg, batch_norm=False):
    layers = []
    in_channels = 3
    activation = ReLU_quant(q_cfg['a_bits']) if q_cfg['activation_quant'] else nn.ReLU()
    fl_bit = q_cfg['w_bits'] if q_cfg['w_bits'] > 3 else 8
    fl_lambda_s = q_cfg['lambda_s']

    if q_cfg['quant_fl']:
        first_layer = Conv2d_minmax(fl_lambda_s, fl_bit, in_channels, cfg[0], kernel_size=3, padding=1, use_fp=q_cfg['use_fp'], bias=False)
    else:
        first_layer = nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=1, bias=False)

    if batch_norm:
        layers += [first_layer, nn.BatchNorm2d(cfg[0]), activation]
    else:
        layers += [first_layer, activation]
    in_channels = cfg[0]

    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d_minmax(q_cfg['lambda_s'], q_cfg['w_bits'], in_channels, v, kernel_size=3, padding=1, use_fp=q_cfg['use_fp'], bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation]
            else:
                layers += [conv2d, activation]
            in_channels = v
    return nn.Sequential(*layers)


def Vgg16_bn(w_bits, a_bits, lambda_s, use_fp=True, activation_quant=True, quant_first_last=False):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    q_cfg = {'w_bits': w_bits, "a_bits": a_bits, "lambda_s": lambda_s, "use_fp": use_fp,\
             "activation_quant": activation_quant, "quant_fl":quant_first_last}
    return VGG(make_layers(cfg['D'], q_cfg, batch_norm=True), q_cfg)