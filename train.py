# Change the dataset path
DATASET_PATH ='~/data'

import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

from logger import SummaryLogger
import utils
import our_network


parser = argparse.ArgumentParser(description='Quantization finetuning for CIFAR100')
parser.add_argument('--arch', metavar='ARCH', default='ResNet32', choices=['ResNet32', 'Vgg16_bn'])
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar100/4bits', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.01', type=float) # By default 1e-4 for Adam // 1e-2 for SGD when starting from EPOCH 82
parser.add_argument('--resume_epoch', default='83', type=int)
parser.add_argument('--epoch', default='150', type=int)
parser.add_argument('--decay_epoch', default=[123], nargs="*", type=int)
parser.add_argument('--w_decay', default='1e-4', type=float)
parser.add_argument('--adam', default='0', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)

parser.add_argument('--load_pretrained', default='models/ResNet82.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)

parser.add_argument('--w_bit', default='4', type=int)
parser.add_argument('--lambda_s', default='150', type=float)
parser.add_argument('--a_bit', default='4', type=float)
parser.add_argument('--first_last_quant', default=1, type=int)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)

torch.manual_seed(int(args.seed))
os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
trainloader, valloader, testloader = utils.get_cifar100_dataloaders(128, 100)

#Quantization parameters
base_lr = args.lr
lambda_s = args.lambda_s
w_bits = args.w_bit
a_bits = args.a_bit
fl_quant = True if args.first_last_quant else False

#Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
USE_ADAM = int(args.adam)
if w_bits == 2:
    print("*" * 20)
    print("W_DECAY set to 0")
    print("*" * 20)
    W_DECAY = 0

model = our_network.__dict__[args.arch](w_bits, a_bits, lambda_s, use_fp=True, activation_quant=False, quant_first_last=fl_quant)

if len(args.load_pretrained) > 2 :
    path = args.load_pretrained
    state = torch.load(path)
    utils.load_checkpoint(model, state)

model.to(DEVICE)

if not USE_ADAM:
    optimizer = optim.SGD(model.parameters(), lr=base_lr, nesterov=False, momentum=0.9, weight_decay=W_DECAY)
else:
    print("*" *20)
    print("Using Adam as optimizer...")
    print("*" *20)
    base_lr *= 1e-2 * lambda_s
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=W_DECAY)
    optimizer.load_state_dict(state['optimizer'])

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion_CE = nn.CrossEntropyLoss()

def eval(net, test_flag=False):
    loader = valloader if not test_flag else testloader
    flag = 'Val.' if not test_flag else 'Test'

    epoch_start_time = time.time()
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)

        loss = criterion_CE(outputs, targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (val_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return val_loss / (b_idx + 1), correct / total

def train(model, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion_CE(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc s1: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1), correct / total

def eval_LP(address, lambda_s,num_bits, test_flag=False):
    net = our_network.__dict__[args.arch](num_bits, a_bits, lambda_s, use_fp=False, activation_quant=False, quant_first_last=fl_quant)

    old_param = torch.load(address)
    net.load_state_dict(old_param)
    net.to(DEVICE)
    if test_flag:
        print("***Test***")
    print("Low Precision: ")
    val_loss, acc = eval(net, test_flag)
    return val_loss, acc

def eval_FP(address, lambda_s,num_bits, test_flag=False):
    net = our_network.__dict__[args.arch](num_bits, a_bits, lambda_s, use_fp=True, activation_quant=False, quant_first_last=fl_quant)

    old_param = torch.load(address)
    net.load_state_dict(old_param)
    net.to(DEVICE)

    if test_flag:
        print("***Test***")
    print("Full Precision: ")
    val_loss, acc = eval(net, test_flag)
    return val_loss, acc

if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'Bit{}_Scale{}_{}'.format(w_bits, lambda_s, time_log)
    else:
        folder_name = 'Bit{}_Scale{}'.format(w_bits, lambda_s)

    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    logger = SummaryLogger(path)
    best_FP = 0
    best_LP = 0

    with open(os.path.join("logs/" + path, 'log.txt'), "a") as f:
        torch.save(model.state_dict(), "ckpt/{}/temp.t7".format(path))
        address = "ckpt/{}/temp.t7".format(path)
        print("Performance of pre-trained model")
        _ , _ = eval_FP(address, lambda_s, w_bits, test_flag=True)
        _ , _ = eval_LP(address, lambda_s, w_bits, test_flag=True)

    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")
        ### Train ###
        train_loss, acc = train(model, epoch)
        scheduler.step()
        ### save for evaluating LP and FP ###
        torch.save(model.state_dict(), "ckpt/{}/temp.t7".format(path))
        address = "ckpt/{}/temp.t7".format(path)
        ### Evaluate LP and FP models ###
        val_loss_LP, accuracy_LP = eval_LP(address,lambda_s,w_bits, test_flag=True)
        val_loss_FP, accuracy_FP = eval_FP(address,lambda_s,w_bits, test_flag=True)

        is_best = accuracy_FP > best_FP
        best_FP = max(accuracy_FP, best_FP)
        LP_is_best = accuracy_LP > best_LP
        best_LP = max(accuracy_LP, best_LP)

        utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_FP_acc': best_FP,
                    'best_LP_acc' : best_LP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, 'ckpt/' + path, filename='{}.pth'.format(epoch))

        train_log = {'Loss': train_loss, 'Accuracy': acc}
        val_log = {'LP loss': val_loss_LP, 'LP accuracy': accuracy_LP,
                    'FP loss': val_loss_FP, 'FP accuracy': accuracy_FP}

        logger.add_scalar_group('Train', train_log, epoch)
        logger.add_scalar_group('Val', val_log, epoch)

        f.write('EPOCH {epoch} \t'
                'Trainacc : {acc:.4f} \t Valacc_LP : {top1_LP:.4f}\t' 
                'Valacc_FP : {top1_FP:.4f} \t Bestacc_LP : {best_LP:.4f} \t' 
                'Bestacc_FP : {best_FP:.4f} \n'.format(
                    epoch=epoch, acc=acc, top1_LP=accuracy_LP, top1_FP=accuracy_FP, best_LP=best_LP, best_FP=best_FP)
                )
        f.close()


    print("*" * 20)
    print("Testing final model")
    print("*" * 20)
    test_loss_LP, test_accuracy_LP = eval_LP(address, lambda_s, w_bits, test_flag=True)
    test_loss_FP, test_accuracy_FP = eval_FP(address, lambda_s, w_bits, test_flag=True)
    f = open(os.path.join("logs/" + path, 'log.txt'), "a")
    f.write('Test FP : {:.4f} \t Test LP : {:.4f}'.format(test_accuracy_FP, test_accuracy_LP))
    f.close()
