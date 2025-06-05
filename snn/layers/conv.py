"""
Spiking Convolution Layers

This module contains convolution layer implementations for spiking neural networks,
including low-latency (LL) convolution layers for both single-step and multi-step
processing modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LLConv2d(nn.Module):
    """Low-latency spiking 2D convolution (single-step)"""
    
    def __init__(self, conv: nn.Conv2d, **kwargs):
        super(LLConv2d, self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level//2 - 1
        self.realize_time = self.steps
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        # self.quan_w_fn = self.conv.quan_w_fn
        
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self, input):
        # print("LLConv2d.steps",self.steps)
        x = input
        
        N, C, H, W = x.shape
        F_h, F_w = self.conv.kernel_size
        S_h, S_w = self.conv.stride
        P_h, P_w = self.conv.padding
        C = self.conv.out_channels
        H = math.floor((H - F_h + 2*P_h)/S_h)+1
        W = math.floor((W - F_w + 2*P_w)/S_w)+1

        if self.zero_output is None:
            # self.zero_output = 0.0
            self.zero_output = torch.zeros(size=(N, C, H, W), device=x.device, dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        # output = self.conv(x)
        if self.realize_time > 0:
            output = torch.nn.functional.conv2d(input, self.conv.weight, (self.conv.bias/self.steps if self.conv.bias is not None else 0.0), stride=self.conv.stride, \
                padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
            self.realize_time = self.realize_time - 1
        else:
            output = torch.nn.functional.conv2d(input, self.conv.weight, None, stride=self.conv.stride, \
                padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

        self.is_work = True
        self.first = False

        return output


class LLConv2d_MS(nn.Module):
    """Low-latency spiking 2D convolution (multi-step)"""
    
    def __init__(self, conv: nn.Conv2d, **kwargs):
        super(LLConv2d_MS, self).__init__()
        self.conv = conv
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = self.level//2 - 1
    
    def forward(self, input):
        # print("LLConv2d_MS input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean())
        # print("LLConv2d_MS input.shape",input.shape)
        B = input.shape[0]//self.T
        
        # # print("LLConv2d_MS.input",input.reshape(torch.Size([self.T,B])+input.shape[1:]).sum(dim=0).abs().mean())
        # # print("LLConv2d_MS.steps",self.steps,"B",B,"self.T",self.T)
        # output = torch.cat([nn.functional.conv2d(input[:B*self.steps], self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups),\
        #                     nn.functional.conv2d(input[B*self.steps:], self.conv.weight, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)], dim=0)
        # # print("LLConv2d_MS.output",output.reshape(torch.Size([self.T,B])+output.shape[1:]).sum(dim=0).abs().mean())
        # # print("LLConv2d_MS output.sum(dim=0).abs().mean()",output.sum(dim=0).abs().mean())
        # # print("LLConv2d_MS output.shape",output.shape)
        # return output
    
        y = F.conv2d(input, self.conv.weight, None,
                    stride=self.conv.stride,
                    padding=self.conv.padding,
                    dilation=self.conv.dilation,
                    groups=self.conv.groups)

        if self.conv.bias is not None and self.steps > 0:
            bias = self.conv.bias.view(1, -1, 1, 1)  
            y[:B * self.steps].add_(bias)
        return y
