'''
Author: Jiayi Liu
Date: 2022-11-29 23:05:30
LastEditors: Jiayi Liu
LastEditTime: 2022-11-30 01:11:21
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''

"""
This is a PyTorch implementation/tutorial of the paper Denoising Diffusion Probabilistic Models.
https://nn.labml.ai/diffusion/ddpm/index.html
"""

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn

from typing import Tuple
from labml_nn.diffusion.ddpm.utils import gather
from labml import monit

class DenoiseDiffusion:

    def __init__(self, eps_model, n_steps, device) -> None:
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        # beta_1 ... beta_T
        self.beta = torch.linspace(1e-4, 2e-2, n_steps).to(device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        self.sigma2 = self.beta 

    def q_xt_x0(self, x0, t) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        q(xt|x0) = N(xt; sqrt(alpha_bar_t)x0, (1-alpha_bar_t)I)
        '''
        mean = gather(self.alpha_bar, t)**0.5*x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var
    
    def q_sample(self, x0, t, eps = None):
        '''
        differentiable
        '''
        if eps == None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean+(var**0.5)*eps

    def p_sample(self, xt, t):
        '''
        p_theta(x_(t-1)|x_t) = N(x_(t-1); mu_theta(x_t, t), sigma^2I)
        mu_theta(x_t, t) = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1-alpha_bar_t) epsilon_theta(x_t, t))
        '''
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1-alpha) / (1-alpha_bar)**0.5
        mean = 1/alpha**0.5 *(xt - eps_coef*eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device = xt.device)
        return mean+var**0.5*eps
    
    def loss(self, x0, noise = None):
        '''
        Loss = E_{t, x0, eps}( || eps - eps_theta(sqrt(alpha_bar_t)x0 + sqrt(1-alpha_bar_t)eps, t) ||^2 )
        Compute the difference between two noises
        '''

        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps = noise)
        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)

    def _sample_x0(self, xt, n_steps):

        for t_ in monit.iterate('Denoise', n_steps):
            t = n_steps - t_ -1
            xt = self.p_sample(xt, t)

        return xt

    def sample(self, image_channels, image_size, device, n_steps, n_samples = 16):
        
        xt = torch.randn([n_samples, image_channels, image_size, image_size], device = device) 
        x0 = self._sample_x0(xt, n_steps)

        for i in range(n_samples):
            self.show_image(x0[i])


        
    
        

        


