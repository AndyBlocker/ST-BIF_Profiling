"""
Quantization Layers and Functions

This module contains quantization-related layers and utility functions
for spiking neural networks, including learnable quantization and 
various quantization helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy


def grad_scale(x, scale):
    """Gradient scaling function for straight-through estimator"""
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def floor_pass(x):
    """Floor function with straight-through gradient"""
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    """Round function with straight-through gradient"""
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def clip(x, eps):
    """Clipping function with gradient preservation"""
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()


def threshold_optimization(data, quantization_level=255, n_trial=300, eps=1e-10):
    '''
    This function collect the activation data and find the optimized clipping
    threshold using KL_div as metric. Since this method is originated from
    post-training quantization which adopted in Tensor-RT, we keep the number of
    bits here.
    Args:
        data(numpy array): activation data
        n_bit(int):
        n_trial(int): the searching steps.
        eps(float): add eps at the average bin step for numberical stability.

    '''

    n_lvl = quantization_level  # quantization levels
    n_half_lvls = (quantization_level)//2
    n_bin_edge = n_lvl * n_trial + 1

    data_max = np.max(np.abs(data))
    hist, bin_edge = np.histogram(data.flatten(),
                                  bins=np.linspace(-data_max,
                                                   data_max,
                                                   num=n_bin_edge))

    mid_idx = int((len(hist)) / 2)
    start_idx = 100
    # log the threshold and corresponding KL-divergence
    kl_result = np.empty([len(range(start_idx, n_trial + 1)), 2])

    for i in range(start_idx, n_trial + 1):
        ref_dist = np.copy(hist[mid_idx - i * n_half_lvls:mid_idx +
                                i * n_half_lvls])
        # merge the outlier
        ref_dist[0] += hist[:mid_idx - i * n_half_lvls].sum()
        ref_dist[-1] += hist[mid_idx + i * n_half_lvls:].sum()
        # perform quantization: bins merge and expansion
        reshape_dim = int(len(ref_dist) / n_lvl)
        ref_dist_reshape = ref_dist.reshape(n_lvl, i)
        # merge bins for quantization
        ref_dist_merged = ref_dist_reshape.sum(axis=1)
        nonzero_mask = (ref_dist_reshape != 0
                        )  # obtain the mask of non-zero bins
        # in each merged large bin, get the average bin_count
        average_bin_count = ref_dist_merged / (nonzero_mask.sum(1) + eps)
        # expand the merged bins back
        expand_bin_count = np.expand_dims(average_bin_count,
                                          axis=1).repeat(i, axis=1)
        candidate_dist = (nonzero_mask * expand_bin_count).flatten()
        kl_div = scipy.stats.entropy(candidate_dist / candidate_dist.sum(),
                                     ref_dist / ref_dist.sum())
        #log threshold and KL-divergence
        current_th = np.abs(
            bin_edge[mid_idx - i * n_half_lvls])  # obtain current threshold
        kl_result[i -
                  start_idx, 0], kl_result[i -
                                           start_idx, 1] = current_th, kl_div

    # based on the logged kl-div result, find the threshold correspond to the smallest kl-div
    th_sel = kl_result[kl_result[:, 1] == kl_result[:, 1].min()][0, 0]
    print(f"Threshold calibration of current layer finished!, calculate threshold {th_sel}")

    return th_sel


class MyQuan(nn.Module):
    """Quantization module with learnable scaling factor"""
    
    def __init__(self, level, sym=False, **kwargs):
        super(MyQuan, self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 512:
            print("level", level)
            self.pos_max = 'full'
        else:
            print("level", level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2 + 1))
            else:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self, name, tfwriter, global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        # print("self.pos_max",self.pos_max)
        if self.pos_max == 'full':
            return x
        # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
        # print("MyQuan intput x.abs().mean()",x.abs().mean(),x.dtype)
        if str(self.neg_min.device) == 'cpu':
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == 'cpu':
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max
        # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
        # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))

        # print("self.init_state",self.init_state)
        # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
        # print("self.training",self.training)
        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max ** 0.5)).cuda() if self.sym \
                            else torch.tensor(x.detach().abs().mean() * 4 / (self.pos_max ** 0.5)).cuda()
            self.init_state += 1
            return x
        # elif self.init_state<self.batch_init and self.training:
        #     self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
        #     self.init_state += 1
        # print("s_grad_scale",s_grad_scale.item(),"self.s",self.s.data.item())

        # elif self.init_state==self.batch_init and self.training:
        #     # self.s = torch.nn.Parameter(self.s)
        #     self.init_state += 1
        #     # print("initialize finish!!!!")
        l2_loss1 = 0
        s_scale = grad_scale(self.s, s_grad_scale)
        # s_scale = s_scale * ((self.level_init)/(self.pos_max))
        output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        # print("MyQuan output.abs().mean()",output.abs().mean(),output.dtype)

        # x_abs = torch.abs(output)/self.s
        # self.l2_loss = l2_loss1 + (x_abs - (1/147)*x_abs*x_abs*x_abs).sum()
        # self.absvalue = (torch.abs(output)/self.s).sum()
        # output = floor_pass(x/s_scale)*s_scale
        return output


class QuanConv2d(torch.nn.Conv2d):
    """Quantized 2D Convolution layer"""
    
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return self._conv_forward(x, quantized_weight, self.bias)


class QuanLinear(torch.nn.Linear):
    """Quantized Linear layer"""
    
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        # self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        return torch.nn.functional.linear(x, quantized_weight, self.bias)