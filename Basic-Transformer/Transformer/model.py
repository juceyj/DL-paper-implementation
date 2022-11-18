'''
Author: Jiayi Liu
Date: 2022-11-09 15:03:46
LastEditors: Jiayi Liu
LastEditTime: 2022-11-09 18:08:14
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from embeddings import TrasnformerEmbedding
from blocks import EncoderBlock, DecoderBlock


class Encoder(nn.Module):

    def __init__(self, n_embedding, d_model, d_hidden, n_block, n_head, max_len, drop_out, device) -> None:
        super().__init__()

        self.emb = TrasnformerEmbedding(
            num_embedding=n_embedding, d_model=d_model, max_len=max_len, drop_prob=drop_out, device=device)
        self.block = nn.ModuleList([EncoderBlock(
            d_model=d_model, d_hidden=d_hidden, n_head=n_head, drop_out=drop_out) for _ in range(n_block)])

    def forward(self, x, s_mask):

        x = self.emb(x)

        for block in self.block:
            x = block(x, s_mask)

        return x


class Decoder(nn.Module):

    def __init__(self, n_embedding, d_model, d_hidden, n_block, n_head, max_len, drop_out, device) -> None:
        super().__init__()

        self.emb = TrasnformerEmbedding(
            num_embedding=n_embedding, d_model=d_model, max_len=max_len, drop_prob=drop_out, device=device)
        self.block = nn.ModuleList([DecoderBlock(
            d_model=d_model, d_hidden=d_hidden, n_head=n_head, drop_out=drop_out) for _ in range(n_block)])
        self.linear = nn.Linear(d_model, n_embedding)

    def forward(self, trg, src, s_mask, t_mask):

        trg = self.emb(trg)

        for block in self.block:
            trg = block(trg, src, s_mask, t_mask)

        return self.linear(trg)

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, src_voc_size, trg_voc_size, d_model, n_head, max_len, d_hidden, n_block, drop_out, device) -> None:
        super().__init__()

        self.src_pad_idx = src_pad_idx # padding index
        self.trg_pad_idx = trg_pad_idx

        self.device = device
        self.encoder = Encoder(n_embedding=src_voc_size, d_hidden=d_hidden, d_model=d_model,
                               n_block=n_block, n_head=n_head, max_len=max_len, drop_out=drop_out, device=device)
        self.decoder = Decoder(n_embedding=trg_voc_size, d_hidden=d_hidden, d_model=d_model,
                               n_block=n_block, n_head=n_head, max_len=max_len, drop_out=drop_out, device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_pad_mask(trg, trg) * self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, trg_mask, src_trg_mask)

        return out

    def make_pad_mask(self, q, k):
        '''
        create padding mask
        '''
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)
        return k&q

    def make_no_peak_mask(self, q, k):
        '''
        create info mask
        '''
        len_q, len_k = q.size(1), k.size(1)
        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask
