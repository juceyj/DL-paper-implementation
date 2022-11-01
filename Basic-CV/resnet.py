'''
Author: Jiayi Liu
Date: 2022-9-31 20:34:25
LastEditors: Jiayi Liu
LastEditTime: 2022-11-01 19:32:25
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

######################
## Resnet in torch ###
######################

import torch
import torch.nn as nn 
import torchinfo

class BasicBlock(nn.Module):
    '''
    Basic block for resnet 18/34
    '''

    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.residual = nn.Sequential(
            # typical 3*3 kernel: (s - k + 2p)/s + 1 -> (s-3+2)/1+1= s 
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size = 3, padding =1, bias = False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )
        
        # use shortcut to match the dimensions of resudual and output
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class BottleNeckBlock(nn.Module):
    '''
    Bottle neck block for resnt 50/101/152
    '''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            # only one 3*3 kernel
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels * BottleNeckBlock.expansion, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeckBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeckBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeckBlock.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * BottleNeckBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace = True)(self.bottleneck(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.conv2 = self._make_layer(block, 64, num_block[0], 2)
        self.conv3 = self._make_layer(block, 128, num_block[1], 2)
        self.conv4 = self._make_layer(block, 256, num_block[2], 2)
        self.conv5 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, first_stride):

        strides = [first_stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
        
if __name__ == '__main__':
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], 100)
    # resnet = ResNet(BottleNeckBlock, [3, 4, 6, 3], 100)
    print(torchinfo.summary(resnet, input_size = (128, 3, 224, 224)))
