"""
Miscellaneous Utilities for Spiking Neural Networks

This module contains miscellaneous utility classes and functions
including element-wise operations, spiking softmax, and legacy attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Addition(nn.Module):
    """Element-wise addition layer for skip connections"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[0] + x[1]


class spiking_softmax(nn.Module):
    """Spiking softmax with temporal accumulation"""
    
    def __init__(self, step, T):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = None
        self.step = step
        self.t = 0
        self.T = T
        # print(self.divide)
    
    def reset(self):
        # print("spiking_softmax reset")
        self.X = 0.0
        self.Y_pre = None       
        self.t = 0
    
    def forward(self, input):
        ori_shape = input.shape
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        input = torch.cumsum(input, dim=0)
        output = F.softmax(input, dim=-1)
        output = torch.diff(output, dim=0, prepend=(output[0]*0.0).unsqueeze(0))
        return output.reshape(ori_shape)


class Attention_no_softmax(nn.Module):
    """Legacy attention implementation without softmax (for compatibility)"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_Relu = nn.ReLU(inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):   
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_Relu(attn)/(N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x