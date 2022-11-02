'''
Author: Jiayi Liu
Date: 2022-11-02 11:41:52
LastEditors: Jiayi Liu
LastEditTime: 2022-11-02 20:41:21
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

"""
[1] Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang
    Residual Attention Network for Image Classification
    https://arxiv.org/abs/1704.06904
"""




import torch
import torch.nn as nn
class ResidualUnit(nn.Module):
    '''
    Bottle neck block for resnt 50/101/152
    1*1 -> 3*3 -> 1*1 -> f(x) + x
    '''
    unit_split = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.bottle_channels = int(out_channels/ResidualUnit.unit_split)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, self.bottle_channels,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(self.bottle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.bottle_channels, self.bottle_channels,
                      kernel_size=3, padiding=1),
            nn.BatchNorm2d(self.bottle_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.bottle_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.bottleneck(x) + self.shortcut(x))


class AttentionModule(nn.Module):

    '''
    T(x) * ( M(x) + 1 )
    '''

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1) -> None:
        super().__init__()

        assert in_channels == out_channels

    def forward(self, x):
        return out
