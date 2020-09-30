# Change the dataset path
DATASET_PATH ='~/data'

import os
import argparse
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
import our_network
import quant_utils

parser = argparse.ArgumentParser(description='Test for CIFAR10/100')
parser.add_argument('--arch', metavar='ARCH', default='ResNet32', choices=['ResNet32', 'Vgg16_bn'])
parser.add_argument('--text', default='result.txt', type=str)
parser.add_argument('--exp_name', default='cifar100', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--model_path', default='models/W4_ResNet.pth', type=str)

parser.add_argument('--w_bit', default='4', type=int)
parser.add_argument('--lambda_s', default='150', type=float) # For logging purposes
parser.add_argument('--a_bit', default='8', type=float)
parser.add_argument('--first_last_quant', default=1, type=int)
parser.add_argument('--act_quant', default=0, type=int)
parser.add_argument('--act_clipping', default=0, type=int)
parser.add_argument('--clipping_range', default=6, type=int)
parser.add_argument('--cu_num', default='0', type=str)

args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

lambda_s = args.lambda_s
w_bits = args.w_bit
a_bits = args.a_bit
act_quant = True if args.act_quant else False
fl_quant = True if args.first_last_quant else False
act_clipping = True if args.act_clipping else False
clipping_range = args.clipping_range
DEVICE = torch.device("cuda")
EXPERIMENT_NAME = args.exp_name

trainloader, valloader, testloader = utils.get_cifar100_dataloaders(128, 100)
model = our_network.ResNet32(w_bits, a_bits, lambda_s, use_fp=True, activation_quant=False, quant_first_last=fl_quant)
lp_net = our_network.ResNet32(w_bits, a_bits, lambda_s, use_fp=False, activation_quant=act_quant, quant_first_last=fl_quant)

states = torch.load(args.model_path, map_location=DEVICE)
utils.load_checkpoint(model, states)
if 'state_dict' in states.keys():
    epoch = states['epoch']
else:
    epoch = 0
model.to(DEVICE)
criterion_CE = nn.CrossEntropyLoss()

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total

def test_LP():
    global my_list
    net = lp_net
    utils.load_checkpoint(net, states)
    if act_quant and act_clipping:
        _ = quant_utils.search_bn(net, clipping_range)
    net.to(DEVICE)
    print("Low Precision: ")
    test_loss, acc = test(net)
    return test_loss, acc

def test_FP():
    net = model
    net.to(DEVICE)
    print("Full Precision: ")
    test_loss, acc = test(net)
    return test_loss, acc

if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'W{}A{}Scale{}_{}'.format(w_bits, a_bits, lambda_s, time_log)
    else:
        folder_name = 'W{}A{}Scale{}'.format(w_bits, a_bits, lambda_s)

    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('results/' + path):
        os.makedirs('results/' + path)
    # Save argparse arguments as logging
    with open('results/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    f = open(os.path.join("results/"+ path, args.text), "a")
    test_loss_LP, accuracy_LP = test_LP()
    test_loss_FP, accuracy_FP = test_FP()

    f.write('{} \t EPOCH {epoch} \t'
            'FP Acc {:.4f} \t LP Acc {:.4f} \n'.format(
                time_log, accuracy_FP, accuracy_LP, epoch=epoch))
    f.close()

