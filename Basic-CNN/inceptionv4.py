'''
Author: Jiayi Liu
Date: 2022-10-01 19:33:23
LastEditors: Jiayi Liu
LastEditTime: 2022-11-02 20:39:06
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

"""
[1] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class InceptionStem(nn.Module):
    '''
    Stem of Inception-v4 and Iception-ResNet-v2: Figure 3 

    (299, 299, 3) -> (35, 35, 384)
    '''

    def __init__(self, in_channels) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            BasicBlock(in_channels, 32, kernel_size=3, stride = 2),
            BasicBlock(32, 32, kernel_size=3),
            BasicBlock(32, 64, kernel_size=3, padding=1)
        )

        self.branch2_conv = BasicBlock(64, 96, kernel_size=3, stride=2)
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch3a = nn.Sequential(
            BasicBlock(160, 64, kernel_size=1),
            BasicBlock(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicBlock(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicBlock(64, 96, kernel_size=3)
        )

        self.branch3b = nn.Sequential(
            BasicBlock(160, 64, kernel_size=1),
            BasicBlock(64, 96, kernel_size=3)
        )

        self.branch4a = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch4b = BasicBlock(192, 192, kernel_size=3, stride = 2)

    def forward(self, x):

        x = self.conv1(x)

        x = [
            self.branch2_conv(x),
            self.branch2_pool(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch3a(x),
            self.branch3b(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch4a(x),
            self.branch4b(x)
        ]

        x = torch.cat(x, 1)

        return x

class InceptionA(nn.Module):
    '''
    Inception-A: figure 4

    (35, 35) -> (35, 35)
    '''
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicBlock(in_channels, 64, kernel_size=1),
            BasicBlock(64, 96, kernel_size=3, padding=1),
            BasicBlock(96, 96, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicBlock(in_channels, 64, kernel_size=1),
            BasicBlock(64, 96, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicBlock(in_channels, 96, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels, 96, kernel_size=1)
        )

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionA(nn.Module):
    '''
    ReductionA: figure 7

    (35, 35) -> (17, 17)
    '''

    def __init__(self, in_channels, k, l, m, n):

        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicBlock(in_channels, k, kernel_size=1),
            BasicBlock(k, l, kernel_size=3, padding=1),
            BasicBlock(l, m, kernel_size=3, stride=2)
        )

        self.branch3x3 = BasicBlock(in_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = in_channels + n + m

    def forward(self, x):

        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionB(nn.Module):
    '''
    Inception-A: figure 5

    (17, 17) -> (17, 17)
    '''
    
    def __init__(self, in_channels):
        super().__init__()

        self.branch7x7stack = nn.Sequential(
            BasicBlock(in_channels, 192, kernel_size=1),
            BasicBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicBlock(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicBlock(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicBlock(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7 = nn.Sequential(
            BasicBlock(in_channels, 192, kernel_size=1),
            BasicBlock(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicBlock(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch1x1 = BasicBlock(in_channels, 384, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicBlock(in_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class ReductionB(nn.Module):
    '''
    ReductionB: figure 6

    (17, 17) -> (8, 8)
    '''

    def __init__(self, in_channels):

        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicBlock(in_channels, 256, kernel_size=1),
            BasicBlock(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicBlock(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicBlock(320, 320, kernel_size=3, stride=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicBlock(in_channels, 192, kernel_size=1),
            BasicBlock(192, 192, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)

class InceptionC(nn.Module):
    '''
    Inception-C: figure 6

    (8, 8) -> (8, 8)
    '''

    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicBlock(in_channels, 384, kernel_size=1),
            BasicBlock(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicBlock(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch3x3stacka = BasicBlock(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicBlock(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicBlock(in_channels, 384, kernel_size=1)
        self.branch3x3a = BasicBlock(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicBlock(384, 256, kernel_size=(1, 3), padding=(0, 1))

        self.branch1x1 = BasicBlock(in_channels, 256, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)

class InceptionV4(nn.Module):

    '''
    stem + SUM(n_i * inception_i + reduction_i) + average_pool + dropout + softmax
    '''

    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=100):

        super().__init__()
        self.stem = InceptionStem(3)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AvgPool2d(7)

        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(in_channels, output_channels, block_num, block):

        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(in_channels))
            in_channels = output_channels

        return layers

class AddResidual(nn.Module):
    '''
    add residual to each inception
    '''

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual):

        x = self.shortcut(x)
        out = self.bn(x + residual)
        out = self.relu(out)

        return out

if __name__ == '__main__':
    pass