'''
Author: Jiayi Liu
Date: 2022-11-02 16:58:43
LastEditors: Jiayi Liu
LastEditTime: 2022-11-03 15:10:21
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

"""
[1] Redmon, Joseph and Farhadi, Ali
    YOLOv3: An Incremental Improvement
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

