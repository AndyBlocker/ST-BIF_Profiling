"""
Spiking Linear Layers

This module contains linear layer implementations for spiking neural networks,
including low-latency (LL) linear layers for both single-step and multi-step
processing modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLLinear(nn.Module):
    """Low-latency spiking linear layer (single-step)"""
    
    def __init__(self, linear, **kwargs):
        super(LLLinear, self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        # self.quan_w_fn = self.linear.quan_w_fn
        
    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self, input):
        # print("LLLinear", input.mean())
        # print("LLLinear.steps",self.steps)
        x = input
        if x.ndim == 2:
            B, N = x.shape
        elif x.ndim == 3:
            B, C, N = x.shape
        N = self.linear.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(size=shape_new, device=x.device, dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.linear.bias.data/self.steps if self.linear.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        # output = self.linear(x)
        if self.realize_time > 0:
            output = torch.nn.functional.linear(input, self.linear.weight, (self.linear.bias/self.steps if self.linear.bias is not None else 0.0))
            self.realize_time = self.realize_time - 1
        else:
            output = torch.nn.functional.linear(input, self.linear.weight, None)

        self.is_work = True
        self.first = False

        return output


class LLLinear_MS(nn.Module):
    """Low-latency spiking linear layer (multi-step)"""
    
    def __init__(self, linear: nn.Linear, **kwargs):
        super(LLLinear_MS, self).__init__()
        self.linear = linear
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = self.level//2 - 1
    
    def forward(self, input):
        B = input.shape[0]//self.T        
        # print("self.steps",self.steps,"B",B,"self.T",self.T)
        output = torch.cat([nn.functional.linear(input[:B*self.steps], self.linear.weight, self.linear.bias),\
                            nn.functional.linear(input[B*self.steps:], self.linear.weight)])
        return output