'''
Author: Jiayi Liu
Date: 2022-11-02 11:41:52
LastEditors: Jiayi Liu
LastEditTime: 2022-11-03 11:10:15
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
import torch.nn.functional as F
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
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.bottle_channels,
                      kernel_size=1, stride=stride, bias=False),

            nn.BatchNorm2d(self.bottle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottle_channels, self.bottle_channels,
                      kernel_size=3, padiding=1, bias=False),

            nn.BatchNorm2d(self.bottle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottle_channels, out_channels,
                      kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.bottleneck(x) + self.shortcut(x)


class AttentionModule1(nn.Module):

    '''
    T(x) * ( M(x) + 1 ): Figure 2
    '''

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1) -> None:
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=p)
        self.trunk = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=t)
        self.mask_down1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down3 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down4 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.mask_up1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up3 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up4 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.short_cut_short = ResidualUnit(
            in_channels=in_channels, out_channels=out_channels, stride=1)
        self.short_cut_long = ResidualUnit(
            in_channels=in_channels, out_channels=out_channels, stride=1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.Sigmoid()
        )
        self.last = self._make_residual(in_channels=in_channels, out_channels=out_channels, p)

    def _make_residual(self, in_channels, out_channels, num_block):

        layers = []
        for _ in range(num_block):
            layers.append(ResidualUnit(in_channels=in_channels,
                          out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 56
        x = self.pre(x)
        shape0 = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        x_m = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
        # 28
        x_m = self.mask_down1(x_m)

        shape1 = (x_m.size(2), x_m.size(3))
        x_skip_connection_long = self.short_cut_long(x_m)

        x_m = F.max_pool2d(x_m, kernel_size=3, padding=1, stride=2)
        # 14
        x_m = self.mask_down2(x_m)

        shape2 = (x_m.size(2), x_m.size(3))
        x_skip_connection_short = self.short_cut_short(x_m)

        x_m = F.max_pool2d(x_m, kernel_size=3, padding=1, stride=2)
        # 7
        x_m = self.mask_down3(x_m)

        x_m = self.mask_down4(x_m)
        x_m = self.mask_up4(x_m)

        x_m = self.mask_up3(x_m)
        x_m = F.interpolate(x_m, size=shape2)
        # 14
        x_m += x_skip_connection_short

        x_m = self.mask_up2(x_m)
        x_m = F.interpolate(x_m, size=shape1)
        # 28
        x_m += x_skip_connection_long

        x_m = self.mask_up1(x_m)
        x_m = F.interpolate(x_m, size=shape0)
        # 56

        x_m = self.sigmoid(x_m)
        out = (x_m+1)*x_t
        out = self.last(out)

        return out


class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1) -> None:
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=p)
        self.trunk = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=t)
        self.mask_down1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down3 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.mask_up1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up3 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.short_cut_short = ResidualUnit(
            in_channels=in_channels, out_channels=out_channels, stride=1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.Sigmoid()
        )
        self.last = self._make_residual(in_channels=in_channels, out_channels=out_channels, p)

    def _make_residual(self, in_channels, out_channels, num_block):

        layers = []
        for _ in range(num_block):
            layers.append(ResidualUnit(in_channels=in_channels,
                          out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 28
        x = self.pre(x)
        shape0 = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        x_m = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
        # 14
        x_m = self.mask_down1(x_m)

        shape1 = (x_m.size(2), x_m.size(3))
        x_skip_connection_short = self.short_cut_short(x_m)

        x_m = F.max_pool2d(x_m, kernel_size=3, padding=1, stride=2)
        # 7
        x_m = self.mask_down2(x_m)

        x_m = self.mask_down3(x_m)
        x_m = self.mask_up3(x_m)

        x_m = self.mask_up2(x_m)
        x_m = F.interpolate(x_m, size=shape1)
        # 14
        x_m += x_skip_connection_short

        x_m = self.mask_up1(x_m)
        x_m = F.interpolate(x_m, size=shape0)
        # 28

        x_m = self.sigmoid(x_m)
        out = (x_m+1)*x_t
        out = self.last(out)

        return out


class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1) -> None:
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=p)
        self.trunk = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=t)
        self.mask_down1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_down2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.mask_up1 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)
        self.mask_up2 = self._make_residual(
            in_channels=in_channels, out_channels=out_channels, num_block=r)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=1),

            nn.Sigmoid()
        )
        self.last = self._make_residual(in_channels=in_channels, out_channels=out_channels, p)

    def _make_residual(self, in_channels, out_channels, num_block):

        layers = []
        for _ in range(num_block):
            layers.append(ResidualUnit(in_channels=in_channels,
                          out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 14
        x = self.pre(x)
        shape0 = (x.size(2), x.size(3))

        x_t = self.trunk(x)

        x_m = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
        # 7
        x_m = self.mask_down1(x_m)

        x_m = self.mask_down2(x_m)
        x_m = self.mask_up2(x_m)

        x_m = self.mask_up1(x_m)
        x_m = F.interpolate(x_m, size=shape0)
        # 14

        x_m = self.sigmoid(x_m)
        out = (x_m+1)*x_t
        out = self.last(out)

        return out


class Attention(nn.Module):
    """
    residual attention netowrk
    """

    def __init__(self, block_num, class_num=100) -> None:
        super().__init__()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # how to keep the same size (56,56) after max-pool?
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 256, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(
            256, 512, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(
            512, 1024, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            ResidualUnit(1024, 2048, 2),
            ResidualUnit(2048, 2048, 1),
            ResidualUnit(2048, 2048, 1)
        )
        self.avg_pool = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.max_pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(ResidualUnit(in_channels, out_channels, 2))

        for _ in range(num):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
