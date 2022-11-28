'''
Author: Jiayi Liu
Date: 2022-11-24 06:02:35
LastEditors: Jiayi Liu
LastEditTime: 2022-11-24 06:17:31
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    '''
    Variational Autoencoder
    '''

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2a = nn.Linear(512, hidden_size)
        self.fc2b = nn.Linear(512, hidden_size)

        self.relu = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(hidden_size, 512)
        self.fc4 = nn.Linear(512, input_size)

    def encode(self, x):
        '''
        return: mu, logvar
        '''
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2a(x), self.fc2b(x)

    def decode(self, z):
        z = self.fc3(z)
        z = self.relu(z)
        return torch.sigmoid(self.fc4(z)) # why sigmoid?

    def sample(self, mu, logvar):
        '''
        mu + std*eps
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar

def vae_loss(x, recon_x, mu, logvar):
    '''
    see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD