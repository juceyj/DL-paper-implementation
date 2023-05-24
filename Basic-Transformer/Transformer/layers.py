'''
Author: Jiayi Liu
Date: 2022-11-09 11:30:14
LastEditors: Jiayi Liu
LastEditTime: 2022-12-08 03:45:45
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None, e=1e12):
    '''
    compute scaled dot product attention.
    '''
    _, _, _, d_k = key.size() # batch_size, head_num, length, key_dim
    key_t = key.transpose(2, 3)
    score = torch.matmul(query, key_t) / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(mask==0,-1e9) # replace 0 with -e
    attn = F.softmax(score, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn

def clones(module, N):
    '''
    generate N same modules
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model, n_head, drop_out=0.1) -> None:
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4) # w_q, w_k, w_v, w_concat
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # same mask applied to all heads
        n_batch = query.size(0)

        # q = w_q(x), k = w_k(x), v = w_v(x) with splited tensor
        # out = softmax(Q@K^T/sqrt(d_k))@V
        query, key, value = [linear(x).view(n_batch, -1, self.n_head, self.d_k).transpose(1,2) for linear, x in zip(self.linears, (query, key, value))]
        print(query.size())
        out, _ = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)
        
        # final linear
        out = out.transpose(1,2).contiguous.view(n_batch, -1, self.d_model)
        return self.linears[-1](out)

class PositionWiseFFN(nn.Module):

    def __init__(self, d_hidden, d_model, drop_out=0.1) -> None:
        super().__init__()
        
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x-mean)/(std+self.eps) * self.gamma + self.beta