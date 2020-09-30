import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import os

plt.style.use('seaborn-ticks')
mpl.use('TkAgg')

paths = {'PSGD': 'models/W4_ResNet.pth' ,'SGD': 'models/ResNet150.pth'}
states = [torch.load(paths['SGD'])['state_dict'], torch.load(paths['PSGD'])['state_dict']]

weight_key = []
for i in states[1].keys():
    if 'weight' in i and ('bn' not in i and 'downsample' not in i) :
        weight_key.append(i)

if not os.path.exists('visualizations/'):
    os.makedirs('visualizations/')

for layer in weight_key:
    fp_tensor = (states[0][layer].flatten().detach().cpu().numpy(), states[1][layer].flatten().detach().cpu().numpy())

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    hist1 = ax1.hist(fp_tensor[0], bins=100, color='b', label='SGD')
    hist2 = ax2.hist(fp_tensor[1], bins=200, color='r', label='PSGD')
    ax1.legend()
    ax2.legend()
    plt.savefig('visualizations/{}.png'.format(layer))
    plt.close()