'''
Author: Jiayi Liu
Date: 2022-10-09 10:24:24
LastEditors: Jiayi Liu
LastEditTime: 2022-11-09 11:27:00
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch 
import torch.nn as nn

class PositionEncoding(nn.Module):
    '''
    Each dimension of the positional encoding corresponds to a sinusoid.
    
    PE_(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE_(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    pos: position
    i:   dimension
    '''
    def __init__(self, d_model, max_len, device) -> None:
        '''
        description: 
        param {*} self
        param {*} d_model: model dimension
        param {*} max_len: max sequence length
        param {*} device: hardware device
        return {*}
        '''
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device) #[512, 512]
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).unsqueeze(dim=1) #[512, 1]
        _2i = torch.arange(0, max_len, step=2, device=device) #[256]

        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model))) #[512, 256]
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
        _, seq_len = x.size() #[128, 12]
        return self.encoding[:seq_len, :] #[12, 512] 

class TokenEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx=1)

class TrasnformerEmbedding(nn.Module):
    '''
    token + positional + dropout
    '''
    def __init__(self, num_embedding, d_model, max_len, drop_prob, device) -> None:
        super().__init__()

        self.tok_emb = TokenEmbedding(num_embeddings=num_embedding, embedding_dim=d_model)
        self.pos_emb = PositionEncoding(d_model=d_model, max_len=max_len, device=device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb) 

