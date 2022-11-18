'''
Author: Jiayi Liu
Date: 2022-11-09 13:12:23
LastEditors: Jiayi Liu
LastEditTime: 2022-11-09 16:43:22
Description: 
Copyright (c) 2022 by JiayiLiu, All Rights Reserved. 
'''
import torch.nn as nn
from layers import MultiHeadedAttention, PositionWiseFFN, LayerNorm

class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_hidden, n_head, drop_out) -> None:
        super().__init__()

        self.attention = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_out)

        self.ffn = PositionWiseFFN(d_hidden=d_hidden, d_model=d_model,drop_out=drop_out)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_out)

    def forward(self, x, mask):
        x_res = x
        # self attention
        x = self.attention(query=x, key=x, value=x, mask=mask)
        
        # add and norm
        x = self.norm1(x + x_res)
        x = self.dropout1(x)

        x_res = x
        # positionwise feed forward network
        x = self.ffn(x)

        # add and norm
        x = self.norm2(x + x_res)
        x = self.dropout2(x)
        return x
    
class DecoderBlock(nn.Module):
        
    def __init__(self, d_model, d_hidden, n_head, drop_out) -> None:
        super().__init__()

        self.attention = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_out)

        self.enc_dec_attn = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_out)

        self.ffn = PositionWiseFFN(d_hidden=d_hidden, d_model=d_model,drop_out=drop_out)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_out)

    def forward(self, enc, dec, s_mask, t_mask):
        dec_res = dec
        # self attention
        dec = self.attention(query=dec , key=dec, value=dec, mask=t_mask)

        # add and norm
        dec = self.norm1(dec+dec_res)
        dec = self.dropout1(dec)

        # enc-dec attention
        if enc is not None:
            dec_res = dec
            dec = self.enc_dec_attn(query=dec, key=enc, value=enc, mask=s_mask)
            
            # add and norm
            dec = self.norm2(dec + dec_res)
            dec = self.dropout2(dec)

        dec_res = dec
        # ffn
        dec = self.ffn(dec)
        # add amd norm
        dec = self.norm3(dec_res + dec)
        dec = self.dropout3(dec)

        return dec

