'''
Autorchor: Jiayi Liu
Date: 2022-10-10 15:50:08
LastEditors: Jiayi Liu
LastEditTime: 2022-11-18 05:07:39
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

import math
import torch
import torch.nn as nn

class LSTM(nn.Module):
    '''
    Implementation of LSTM. 

    a^t = tanh(W_c * x^t + U_c * h^(t-1)) = tanh(hat(a)^t)
    i^t = sigmoid(W_i * x^t + U_i * h^(t-1)) = sigmoid(hat(i)^t)
    f^t = sigmoid(W_f * x^t + U_f * h^(t-1)) = sigmoid(hat(f)^t)
    o^t = sigmoid(W_o * x^t + U_o * h^(t-1)) = sigmoid(hat(o)^t)
    c^t = f^t * c^(t-1) + i^t * a^t
    h^t = o^t * tanh(c^t)
    '''

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        h_t = torch.mul(o_t, c_t.tanh())

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)