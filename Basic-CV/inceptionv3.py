'''
Author: Jiayi Liu
Date: 2022-10-01 19:33:10
LastEditors: Jiayi Liu
LastEditTime: 2022-11-02 11:40:28
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

"""
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567v3
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


class Inception1(nn.Module):
    '''
    naive module

    x -> 1*1 -> x1(same)
    x -> 1*1 -> 5*5 -> x2(same)
    x -> 1*1 -> 3*3 -> 3*3 -> x3(same)
    x -> pool -> 1*1 -> x4(same)

    concatenate[x1, x2, x3, x4] -> [batch, cat(), height, width]
    '''

    def __init__(self, in_channels, pool_channels) -> None:
        super().__init__()

        self.branch1 = BasicBlock(in_channels, 64, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicBlock(in_channels=in_channels,
                       out_channels=48, kernel_size=1),
            BasicBlock(48, 64, kernel_size=5, padding=2)
        )

        self.branch3 = nn.Sequential(
            BasicBlock(in_channels=in_channels,
                       out_channels=64, kernel_size=1),
            BasicBlock(64, 96, kernel_size=3, padding=1),
            BasicBlock(96, 96, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
            BasicBlock(in_channels=in_channels,
                       out_channels=pool_channels, kernel_size=1)
        )

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], 1)


class Inception2(nn.Module):
    '''
    downsample: Figure 10

    x -> 1*1 -> 3*3 (stride 2) -> x1
    x -> 1*1 -> 3*3 (stride 1) -> 3*3 (stride 2) -> x2
    x -> pool (stride 2) -> x3
    '''

    def __init__(self, in_channels) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            BasicBlock(in_channels, 64, kernel_size=1),
            BasicBlock(64, 64, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicBlock(in_channels, 64, kernel_size=1),
            BasicBlock(64, 96, kernel_size=3, padding=1, stride=1),
            BasicBlock(96, 96, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        return torch.cat([x1, x2, x3], 1)


class Inception3(nn.Module):
    '''
    factorization: Figure 6

    x -> 1*1 -> x1(same)
    x -> 1*1 -> 1*n -> n*1 -> x2(same)
    x -> 1*1 -> 1*n -> n*1 -> 1*n -> n*1 -> x3(same)
    x -> pool -> 1*1 -> x4(same)
    '''

    def __init__(self, in_channels, out_channels, factor=7) -> None:
        super().__init__()

        padding = int((factor - 1)/2)

        self.branch1 = BasicBlock(in_channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicBlock(in_channels, out_channels, kernel_size=1),
            BasicBlock(out_channels, out_channels, kernel_size=(
                factor, 1), padding=(padding, 0)),
            BasicBlock(out_channels, 192, kernel_size=(
                1, factor), padding=(0, padding))
        )

        self.branch3 = nn.Sequential(
            BasicBlock(in_channels, out_channels, kernel_size=1),
            BasicBlock(out_channels, out_channels, kernel_size=(
                factor, 1), padding=(padding, 0)),
            BasicBlock(out_channels, out_channels, kernel_size=(
                1, factor), padding=(0, padding)),
            BasicBlock(out_channels, out_channels, kernel_size=(
                factor, 1), padding=(padding, 0)),
            BasicBlock(out_channels, 192, kernel_size=(
                1, factor), padding=(0, padding))
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels, 192, kernel_size=1),
        )

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], 1)


class Inception4(nn.Module):
    '''
    Filter expanded: Figure 7

    x -> 1x1 -> x1(same)

    x -> 1x1 -> 3x1
    x -> 1x1 -> 1x3
    concatenate(3x1, 1x3) -> x2(same)

    x -> 1x1 -> 3x3 -> 1x3
    x -> 1x1 -> 3x3 -> 3x1
    concatenate(1x3, 3x1) -> x3(same)

    x -> pool -> 1*1 -> x4(same)
    '''

    def __init__(self, in_channels) -> None:
        super().__init__()

        self.branch1 = BasicBlock(in_channels, 320, kernel_size=1)

        self.branch2_1 = BasicBlock(in_channels, 384, kernel_size=1)
        self.branch2_2a = BasicBlock(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_2b = BasicBlock(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3_1 = BasicBlock(in_channels, 448, kernel_size=1)
        self.branch3_2 = BasicBlock(448, 384, kernel_size=3, padding=1)
        self.branch3_3a = BasicBlock(
            384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_3b = BasicBlock(
            384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):

        x1 = self.branch1(x)

        x2 = self.branch2_1(x)
        x2 = [
            self.branch2_2a(x2),
            self.branch2_2b(x2)
        ]
        x2 = torch.cat(x2, 1)

        x3 = self.branch3_1(x)
        x3 = self.branch3_2(x3)
        x3 = [
            self.branch3_3a(x3),
            self.branch3_3b(x3)
        ]
        x3 = torch.cat(x3, 1)

        x4 = self.branch_pool(x)

        return torch.cat([x1, x2, x3, x4], 1)

class InceptionV3(nn.Module):
    '''
    the detailed structure is omitted.
    '''

    def __init__(self, num_classes) -> None:
        super().__init__()
        
        self.branch = nn.Sequential()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d()
        self.fc = nn.Linear(2048, num_classes)
        pass

    def forward(self, x):
        x = self.branch(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    pass