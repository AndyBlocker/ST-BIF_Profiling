"""
Spiking Neural Network Normalization Layers

This module contains various normalization layers adapted for spiking neural networks,
including batch normalization and layer normalization variants for temporal processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple


class spiking_BatchNorm2d_MS(nn.Module):
    """Multi-step spiking batch normalization"""
    
    def __init__(self, bn: nn.BatchNorm1d, level, input_allcate=False):
        super(spiking_BatchNorm2d_MS, self).__init__()
        self.level = level
        self.bn = bn
        self.bn.running_mean.requires_grad = False
        self.bn.running_var.requires_grad = False

    def forward(self, input):
        P = input.shape[0]
        # x = input.reshape(torch.Size([int((self.level)),P//int((self.level))]) + input.shape[1:]).detach()
        # self.bn(x.sum(dim=0))

        # print("before BatchNorm",input.reshape(4,32,384,197).sum(dim=0).abs().mean())
        input_shape = len(input.shape)
        if input_shape == 4:
            B, H, N, C = input.shape
            input = input.reshape(B*H, N, C)
        if input_shape == 2:
            input = input.unsqueeze(1)
        input = input.transpose(-1, -2)
        # output = self.bn(input)
        output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level, running_var=self.bn.running_var.detach(), weight=self.bn.weight, bias=self.bn.bias/self.level, eps=self.bn.eps)
        output = output.transpose(-1, -2)
        # print(input.mean(),output.mean())
        if input_shape == 2:
            output = output.squeeze(1)
        if input_shape == 4:
            output = output.reshape(B, H, N, C)
        
        # print("after BatchNorm",output.reshape(4,32,384,197).detach().sum(dim=0).abs().mean())
        return output
        
        
class spiking_BatchNorm2d(nn.Module):
    """Single-step spiking batch normalization with optional input allocation"""
    
    def __init__(self, bn: nn.BatchNorm2d, level, input_allcate):
        super(spiking_BatchNorm2d, self).__init__()
        self.level = level
        self.fire_time = 0
        self.bn = bn
        self.bn.momentum = None
        self.input_allcate = input_allcate
        if self.input_allcate:
            self.input_allocator = torch.nn.Parameter(torch.ones(torch.Size([self.level-1, self.bn.running_mean.shape[0]]))/self.level, requires_grad=True)
        self.eps = bn.eps
        self.accu_input = 0.0
    
    def reset(self):
        # print("spiking_BatchNorm2d reset")
        self.fire_time = 0
        self.accu_input = 0.0
    
    def forward(self, input):
        self.bn.eval()
        self.accu_input = self.accu_input + input
        # print("self.bn.running_mean",self.bn.running_mean.mean())
        # print("self.fire_time",self.fire_time,"self.level",self.level,"self.bn.running_mean",self.bn.running_mean.mean().item(),"self.bn.running_var",self.bn.running_var.mean().item(),"self.bn.weight",self.bn.weight.mean().item(),"self.bn.bias",self.bn.bias.mean().item())
        if self.fire_time < self.level - 1:
            # self.bn.running_mean.data = self.bn.running_mean*self.input_allocator[self.fire_time].detach()
            # self.bn.bias.data = self.bn.bias*self.input_allocator[self.fire_time]
            input_shape = len(input.shape)
            if input_shape == 4:
                B, H, N, C = input.shape
                input = input.reshape(B*H, N, C)
            if input_shape == 2:
                input = input.unsqueeze(1)
            if self.input_allcate:
                input = input.transpose(-1, -2)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()*self.input_allocator[self.fire_time].detach(), running_var=self.bn.running_var, weight=self.bn.weight, bias=self.bn.bias*self.input_allocator[self.fire_time], training=False, eps=self.eps)
                output = output.transpose(-1, -2)
            else:
                input = input.transpose(-1, -2)
                output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach()/self.level, running_var=self.bn.running_var, weight=self.bn.weight, bias=self.bn.bias/self.level, training=False, eps=self.eps)
                output = output.transpose(-1, -2)
            if input_shape == 2:
                output = output.squeeze(1)
            if input_shape == 4:
                output = output.reshape(B, H, N, C)
            self.fire_time = self.fire_time + 1
            return output
        else:
            input_shape = len(input.shape)
            if input_shape == 4:
                B, H, N, C = input.shape
                input = input.reshape(B*H, N, C)
            if input_shape == 2:
                input = input.unsqueeze(1)
            input = input.transpose(-1, -2)
            output = F.batch_norm(input=input, running_mean=self.bn.running_mean.detach(), running_var=self.bn.running_var, weight=self.bn.weight, bias=None, training=False, eps=self.eps)
            output = output.transpose(-1, -2)
            if input_shape == 2:
                output = output.squeeze(1)
            if input_shape == 4:
                output = output.reshape(B, H, N, C)
            self.fire_time = self.fire_time + 1
            return output


class Spiking_LayerNorm(nn.Module):
    """Spiking layer normalization with temporal accumulation"""
    
    def __init__(self, dim, T):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = None
        self.X = 0.0
        self.Y_pre = None
        self.weight = None
        self.bias = None
        self.T = T
        self.t = 0

    def reset(self):
        # print("Spiking_LayerNorm reset")
        self.X = 0.0
        self.Y_pre = None
        self.t = 0
        
    def forward(self, input):
        output = []
        input = input.reshape(torch.Size([self.T, input.shape[0]//self.T]) + input.shape[1:])
        # print("input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(), "after layernorm:", self.layernorm(input.sum(dim=0)).abs().mean())
        for t in range(self.T):
            self.X = input[t] + self.X
            if t < 4:
                Y = self.layernorm(self.X)*((t+1)/4)
            else:
                Y = self.layernorm(self.X)
            if self.Y_pre is not None:
                Y_pre = self.Y_pre.detach().clone()
            else:
                Y_pre = 0.0
            self.Y_pre = Y
            output.append(Y - Y_pre)
        return torch.cat(output, dim=0)


class SpikingBatchNorm2d_MS(nn.Module):
    """Spiking Batch Normalization for multi-step processing"""
    
    def __init__(self, bn, **kwargs):
        super(SpikingBatchNorm2d_MS, self).__init__()
        self.bn = bn  # 保存原始BatchNorm模块
        self.level = kwargs["level"]
        self.T = kwargs["time_step"]
        self.steps = self.level//2 - 1
        self.spike = True
        self.step = kwargs.get("step", None)
    
    def forward(self, input):
        B = input.shape[0] // self.T  # 批次大小
        
        # 1) 一次 BN，不加 bias
        y = F.batch_norm(
            input,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            None,                 # bias 先留空
            self.training,
            self.bn.momentum,
            self.bn.eps
        )

        # 2) 前半段加回原 bias
        if self.steps > 0 and self.bn.bias is not None:
            y[:B*self.steps].add_(self.bn.bias.view(1, -1, 1, 1))

        # 3) 后半段补偿 +μw/σ
        if B*self.steps < input.shape[0]:
            std = torch.sqrt(self.bn.running_var + self.bn.eps)
            comp = (self.bn.weight * self.bn.running_mean) / std        # shape [C]
            y[B*self.steps:].add_(comp.view(1, -1, 1, 1))

        return y
        
        # 第一部分：应用带有缩放过的bias的BatchNorm
        # first_part = F.batch_norm(
        #     input[:B*self.steps],
        #     self.bn.running_mean,  
        #     self.bn.running_var,
        #     self.bn.weight,
        #     self.bn.bias,
        #     self.training,  # 使用当前模块的training状态，而不是self.bn.training
        #     self.bn.momentum,
        #     self.bn.eps
        # ) if B*self.steps > 0 else input.new_tensor([])
        
        # # 第二部分：应用没有bias的BatchNorm
        # second_part = F.batch_norm(
        #     input[B*self.steps:],
        #     torch.zeros_like(self.bn.running_mean).to(input.device),
        #     self.bn.running_var,
        #     self.bn.weight,
        #     None,  # 无bias
        #     self.training,  # 使用当前模块的training状态，而不是self.bn.training
        #     self.bn.momentum,
        #     self.bn.eps
        # ) if input.shape[0] > B*self.steps else input.new_tensor([])
        
        # # 拼接结果
        # if first_part.numel() > 0 and second_part.numel() > 0:
        #     return torch.cat([first_part, second_part], dim=0)
        # elif first_part.numel() > 0:
        #     return first_part
        # else:
        #     return second_part


class MyBatchNorm1d(nn.BatchNorm1d):
    """Custom 1D batch normalization for spiking networks"""
    
    def __init__(self, dim, **kwargs):
        super(MyBatchNorm1d, self).__init__(dim, **kwargs)
        self.spike = False
        self.T = 0
        self.step = 0
        self.momentum = 0.1
    
    def forward(self, x):
        # self.training = False
        input_shape = len(x.shape)
        if input_shape == 4:
            B, H, N, C = x.shape
            x = x.reshape(B*H, N, C)
        if input_shape == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        # if self.spike:
        #     print("before mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("before mybatchnorm1d:",x.abs().mean())
        if not self.spike:
            x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        else:
            Fd = x.shape[0]
            if self.step >= self.T:
                x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)
            else:
                x = torch.cat([F.batch_norm(x[:int(Fd*(self.step/self.T))], self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps), \
                            F.batch_norm(x[int(Fd*(self.step/self.T)):], torch.zeros_like(self.running_mean), self.running_var, self.weight, torch.zeros_like(self.bias), False, self.momentum, self.eps)])
        # if self.spike:
        #     print("after mybatchnorm1d:",x.reshape(torch.Size([self.T,x.shape[0]//self.T]) + x.shape[1:]).sum(dim=0).abs().mean())
        # else:
        #     print("after mybatchnorm1d:",x.abs().mean())
        x = x.transpose(1, 2)
        if input_shape == 2:
            x = x.squeeze(1)
        if input_shape == 4:
            x = x.reshape(B, H, N, C)
        # print("self.running_mean",self.running_mean.abs().mean())
        return x


class LN2BNorm(nn.Module):
    """Hybrid layer norm to batch norm conversion"""
    
    def __init__(self, dim, eps=1e-6):
        super(LN2BNorm, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        self.Eta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("running_mean", torch.zeros(self.dim))
        self.register_buffer("running_var", torch.ones(self.dim))
        self.Lambda = 1.0
        self.momentum = 0.1
        self.eps = eps
        
    def forward(self, x):
        out_LN = F.layer_norm(x, (self.dim,), self.weight, self.bias)
        out_Identity = x + 0.0
        input_shape = len(x.shape)
        if input_shape == 4:
            B, H, N, C = x.shape
            x = x.reshape(B*H, N, C)
        if input_shape == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out_BN = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        out_BN = out_BN.transpose(1, 2)
        if input_shape == 2:
            out_BN = out_BN.squeeze(1)
        if input_shape == 4:
            out_BN = out_BN.reshape(B, H, N, C)
        return out_LN*self.Lambda + out_BN*(1 - self.Lambda)


class MLP_BN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks with batch norm"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=MyBatchNorm1d, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = norm_layer(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MyBachNorm(nn.Module):
    """Temporal batch normalization wrapper"""
    
    def __init__(self, bn, T):
        super(MyBachNorm, self).__init__()
        # bn.bias.data = bn.bias/T
        # bn.running_mean = bn.running_mean/T
        self.bn = bn
        self.T = T
        self.t = 0
    
    def forward(self, x):
        self.bn.eval()
        if self.t == 0:
            self.bn.train()
        self.t = self.t + 1
        if self.t == self.T:
            self.t = 0
        x = self.bn(x)
        return x
    

class MyLayerNorm(nn.Module):
    """Custom layer normalization implementation"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))
        nn.init.constant_(self.weight, 1.)
        nn.init.constant_(self.bias, 0.)
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9
        self.eps = 1e-6
    
    def forward(self, x):        
        if self.training:
            if self.running_mean is None:
                self.running_mean = nn.Parameter((1-self.momentum) * x.mean([-1], keepdim=True), requires_grad=False)
                self.running_var = nn.Parameter((1-self.momentum) * x.var([-1], keepdim=True), requires_grad=False)
            else:
                self.running_mean.data = (1-self.momentum) * x.mean([-1], keepdim=True) + self.momentum * self.running_mean # mean: [1, max_len, 1]
                self.running_var.data = (1-self.momentum) * x.var([-1], keepdim=True) + self.momentum * self.running_var # std: [1, max_len, 1]
            return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias
        else:
            # if self.running_mean is None:
            self.running_mean = nn.Parameter(x.mean([-1], keepdim=True), requires_grad=False)
            self.running_var = nn.Parameter(x.var([-1], keepdim=True), requires_grad=False)
            return self.weight * (x - self.running_mean) / (self.running_var + self.eps) + self.bias