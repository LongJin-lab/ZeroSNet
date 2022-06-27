import torch.nn as nn

import torch
import torch.nn.functional as functional
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron
# from init import *
from random import random
import argparse




__all__ = [ 'zerosnet18_in', 'zerosnet34_in', 'zerosnet50_in', 'zerosnet101_in', 'zerosnet152_in', 'pre_act_resnet18_in', 'pre_act_resnet34_in', 'pre_act_resnet50_in', 'pre_act_resnet101_in', 'pre_act_resnet152_in']


global num_cla
num_cla = 1000




class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion*planes: # "expansion*planes" is the real output channel number
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = functional.relu(self.bn1(x))

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)
        out = out + shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = functional.relu(self.bn1(x))
        input_out = out

        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        out = functional.relu(out)
        out = self.conv3(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
        else:
            shortcut = x

        out = out + shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset="CIFAR10"):
        super(PreActResNet, self).__init__()

        self.in_planes = 64
        self.dataset = dataset

        if dataset == "CIFAR10":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_classes = 10
        elif dataset == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000


        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if dataset == "CIFAR10":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "ImageNet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # print('self.in_planes', self.in_planes)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.dataset == "ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR10":
            out = self.linear(out)
        elif self.dataset == "ImageNet":
            out = self.fc(out)
        return out#, 9999, 999

class ZeroSBlock_IN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, k_ini=-9.0/5,  share_k=False, stepsize=1, given_coe=None, downsample=None,                  pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):
        super(ZeroSBlock_IN,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.share_k = share_k
        self.given_coe = given_coe
        if self.given_coe is not None:
            self.k = k_ini
            self.a_0 = float(given_coe[0])
            self.a_1 = float(given_coe[1])
            self.a_2 = float(given_coe[2])
            self.b_0 = float(given_coe[3])
        elif self.share_k is True:
            self.k = k_ini
        else:
            self.k =nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))

        if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
        # if 1:
            if self.in_planes != self.expansion*planes:
                self.shortcut_x = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
            if self.last_res_planes != self.expansion*planes:
                self.shortcut_l = nn.Sequential(
                    nn.Conv2d(last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
            if self.l_last_res_planes != self.expansion*planes:
                self.shortcut_ll = nn.Sequential(
                    nn.Conv2d(l_last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, inp):

        x = inp[0]
        last_res = inp[1]
        l_last_res = inp[2]

        F_x_n = functional.relu(self.bn1(x))
        if hasattr(self, 'shortcut_x'):
            residual = self.shortcut_x(F_x_n)
        else:
            residual = x

        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = functional.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)

        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):

            if hasattr(self, 'shortcut_l'):
                last_res = self.shortcut_l(last_res)
            if hasattr(self, 'shortcut_ll'):
                l_last_res = self.shortcut_ll(l_last_res)

            if self.given_coe is None: # trainable k
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)

            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
        else:
            x = F_x_n+residual

        l_last_res = last_res
        last_res = residual

        out = [x]+ [last_res]+ [l_last_res]+ [self.k]

        return out

class ZeroSBottleneck_IN(nn.Module): #actually, this is the preact block
    expansion = 4
    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, k_ini=-9.0/5, share_k=False, stepsize=1, given_coe=None, downsample=None):
        super(ZeroSBottleneck_IN,self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # self.bn3 = nn.BatchNorm2d(planes)# 20210803
        self.stride=stride
        self.expansion = 4
        self.in_planes=in_planes
        self.planes=planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.share_k = share_k
        self.given_coe = given_coe
        if self.given_coe is not None:
            self.k = k_ini
            self.a_0 = float(given_coe[0])
            self.a_1 = float(given_coe[1])
            self.a_2 = float(given_coe[2])
            self.b_0 = float(given_coe[3])
        elif self.share_k:
            self.k = k_ini
        else:
            self.k =nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        if self.in_planes != self.expansion*planes:
            self.shortcut_x = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

        if not self.last_res_planes == -1:
            if self.last_res_planes != self.expansion*planes:
                self.shortcut_l = nn.Sequential(
                    nn.Conv2d(last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
        if not self.l_last_res_planes == -1:
            if self.l_last_res_planes != self.expansion*planes:
                self.shortcut_ll = nn.Sequential(
                    nn.Conv2d(l_last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )


    def forward(self, inp):

        x = inp[0]
        last_res = inp[1]
        l_last_res = inp[2]

        F_x_n = functional.relu(self.bn1(x))
        residual = F_x_n

        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = functional.relu(F_x_n)

        F_x_n = self.conv2(F_x_n)
        F_x_n = self.bn3(F_x_n)

        F_x_n = functional.relu(F_x_n)
        F_x_n = self.conv3(F_x_n)
        if hasattr(self, 'shortcut_x'):
            residual = self.shortcut_x(residual)
        else:
            residual = x

        if hasattr(self, 'shortcut_l'):
            last_res = self.shortcut_l(last_res)
        if hasattr(self, 'shortcut_ll'):
            l_last_res = self.shortcut_ll(l_last_res)
        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):

            if self.given_coe is None:
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)

            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)

        else:
            x = F_x_n+residual
        l_last_res = last_res
        last_res = residual

        out = [x]+ [last_res]+ [l_last_res]+ [self.k]
        return out

class zerosnet_in(nn.Module):
    def __init__(self, block, num_blocks, dataset="CIFAR10", k_ini=-9.0 / 5, share_k=False, given_coe=None,
                 pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):

        super(zerosnet_in, self).__init__()
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.noise = noise
        self.block = block
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.share_k = share_k
        self.given_coe = given_coe
        self.stepsize = 1
        # self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))

        self.ks = []
        self.l = 0

        self.in_planes = 64
        self.dataset = dataset
        if self.share_k:
            self.k_ini = nn.Parameter(torch.Tensor(1).uniform_(-9.0 / 5, -9.0 / 5))
        if dataset == "CIFAR10":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_classes = 10
        elif dataset == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if dataset == "CIFAR10":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "ImageNet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            # print('self.l: ', self.l)
            layers.append(block(self.in_planes, planes, self.last_res_planes, self.l_last_res_planes, stride,
                          k_ini=self.k_ini, stepsize=self.stepsize, share_k=self.share_k, given_coe=self.given_coe))

            self.l_last_res_planes = planes * block.expansion
            self.last_res_planes = planes * block.expansion
            self.in_planes = planes * block.expansion
            self.l += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        self.ks = []
        last_res = -1
        l_last_res = -1

        out = self.conv1(x)
        if self.dataset == "ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
        out = [out]+[last_res]+[l_last_res]
        out = self.layer1(out)
        self.ks = self.ks+[out[3]]

        out = self.layer2(out)
        self.ks = self.ks+[out[3]]

        out = self.layer3(out)
        self.ks = self.ks+[out[3]]

        out = self.layer4(out)
        self.ks = self.ks+[out[3]]
        out = out[0]
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR10":
            out = self.linear(out)
        elif self.dataset == "ImageNet":
            out = self.fc(out)
        return out, self.ks, self.stepsize
# coes = [0.3333333, 0.5555556, 0.1111111, 1.77777778]
def zerosnet18_cifar(dataset="CIFAR10", **kwargs):
    return zerosnet_in(ZeroSBlock_IN, [2,2,2,2], dataset=dataset, **kwargs)
def zerosnet18_in(dataset="ImageNet", **kwargs):
    return zerosnet_in(ZeroSBlock_IN, [2,2,2,2], dataset=dataset,**kwargs)
def zerosnet34_cifar(dataset="CIFAR10", **kwargs):
    return zerosnet_in(ZeroSBlock_IN, [3,4,6,3], dataset=dataset, **kwargs)
def zerosnet34_in(dataset="ImageNet", **kwargs):
    return zerosnet_in(ZeroSBlock_IN, [3,4,6,3], dataset=dataset, **kwargs)
def zerosnet50_cifar(dataset = "CIFAR10", **kwargs):
    return zerosnet_in(ZeroSBottleneck_IN, [3,4,6,3], dataset = dataset, **kwargs)
def zerosnet50_in(dataset = "ImageNet", **kwargs):
    return zerosnet_in(ZeroSBottleneck_IN, [3,4,6,3], dataset = dataset, **kwargs)
def zerosnet101_in(dataset = "ImageNet", **kwargs):
    return zerosnet_in(ZeroSBottleneck_IN, [3,4,23,3], dataset = dataset, **kwargs)
def zerosnet152_in(dataset = "ImageNet", **kwargs):
    return zerosnet_in(ZeroSBottleneck_IN, [3,8,36,3], dataset = dataset, **kwargs)

def PreActResNet18_cifar(dataset="CIFAR10"):
    return PreActResNet(PreActBlock, [2,2,2,2], dataset=dataset)
def pre_act_resnet18_in(dataset="ImageNet"):
    return PreActResNet(PreActBlock, [2,2,2,2], dataset=dataset)
def PreActResNet34_cifar(dataset="CIFAR10"):
    return PreActResNet(PreActBlock, [3,4,6,3], dataset=dataset)
def pre_act_resnet34_in(dataset="ImageNet"):
    return PreActResNet(PreActBlock, [3,4,6,3], dataset=dataset)
def PreActResNet50_cifar(dataset = "CIFAR10"):
    return PreActResNet(PreActBottleneck, [3,4,6,3], dataset = dataset)
def pre_act_resnet50_in(dataset = "ImageNet"):
    return PreActResNet(PreActBottleneck, [3,4,6,3], dataset = dataset)
def pre_act_resnet101_in(dataset = "ImageNet"):
    return PreActResNet(PreActBottleneck, [3,4,23,3] , dataset = dataset)
def pre_act_resnet152_in(dataset = "ImageNet"):
    return PreActResNet(PreActBottleneck, [3,8,36,3], dataset = dataset)



if __name__ == '__main__':

    d = torch.randn(2, 3, 224, 224)
    net = zerosnet18_in()
    o = net(d)
    onnx_path = "onnx_model_name_zerosnet18_in.onnx"
    torch.onnx.export(net, d, onnx_path)
    netron.start(onnx_path)

