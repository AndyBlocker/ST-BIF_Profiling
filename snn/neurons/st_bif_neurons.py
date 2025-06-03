"""
ST-BIF (Spike Threshold - Bifurcation) Neuron Models

This module contains ST-BIF neuron implementations with both single-step (SS)
and multi-step (MS) variants, including CUDA-accelerated implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA


def theta_backward(x):
    """Backward pass function for theta (step function) with sigmoid approximation"""
    sigmoid = torch.sigmoid(4*x)
    return 4*sigmoid*(1-sigmoid)
    # tanh = F.tanh(2*x)
    # return 1/(1+(2*x)*(2*x))
    # return 1 - F.tanh(2*x)*F.tanh(2*x)


def theta(x):
    """Step function: returns 1 if x > 0, 0 otherwise"""
    # return (x > 0).int()
    return 1.0*(torch.gt(x,0))
 

def theta_eq(x):
    """Step function: returns 1 if x >= 0, 0 otherwise"""
    # return (x >= 0).int()
    return 1.0*(torch.ge(x,0))


class ST_BIFNodeATGF_SS(torch.autograd.Function):
    """Single-step ST-BIF Node Autograd Function"""
    
    @staticmethod
    def forward(ctx, x_t: torch.Tensor, V_t_1: torch.Tensor, T_t_1: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, t: torch.Tensor):

        spike = x_t * 0.0
        H_t = V_t_1 + x_t
        # spike[torch.logical_and((torch.ge(H_t-v_th,0)), (torch.lt(T_t_1-T_max,0)))] = 1
        # spike[torch.logical_and((torch.lt(H_t,0)), (torch.gt(T_t_1-T_min,0)))] = -1

        spike_condition = (H_t >= v_th) & (T_t_1-T_max < 0)
        neg_spike_condition = (H_t < 0) & (T_t_1-T_min > 0)

        spike = torch.where(spike_condition, torch.ones_like(H_t),
                                      torch.where(neg_spike_condition, -torch.ones_like(H_t),
                                                  torch.zeros_like(H_t)))

        V_t = H_t - v_th * spike
        T_t = T_t_1 + spike

        ctx.save_for_backward(T_t_1,H_t,v_th,T_max,T_min,t)
        
        return spike, V_t, T_t

    @staticmethod
    def backward(ctx, grad_spike_t: torch.Tensor, grad_v_t: torch.Tensor, grad_T_t: torch.Tensor):

        T_t_1,H_t,v_th,T_max,T_min,t = ctx.saved_tensors
        
        grad_T_t_to_H_t = (theta_backward(H_t - v_th)*theta(T_max - T_t_1)+theta_backward(-H_t)*theta(T_t_1 - T_min))
        grad_Y_t_to_T_t_1 = -(theta_eq(H_t-v_th)*theta_backward(T_max - T_t_1)+theta(-H_t)*theta_backward(T_t_1 - T_min))
        
        tmp = grad_spike_t - v_th*grad_v_t + grad_T_t
        grad_X_t = tmp*grad_T_t_to_H_t + grad_v_t
        grad_T_t_1 = tmp*grad_Y_t_to_T_t_1 + grad_T_t
        grad_V_t_1 = grad_X_t + 0.0
        # print("t:",t,"grad_V_t_1",grad_X_t.mean().item(),"grad_T_t_1",grad_T_t_1.mean().item(),"grad_v_t",grad_v_t.mean().item(),"grad_T_t",grad_T_t.mean().item())
        return grad_X_t, grad_V_t_1, grad_T_t_1, None, None, None, None


class ST_BIFNeuron_SS(nn.Module):
    """Single-step ST-BIF Neuron implementation"""
    
    def __init__(self, q_threshold, level, sym=False):
        super(ST_BIFNeuron_SS, self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.T = self.level//2 - 1
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 + 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.init = True
        # self.steps = max(self.pos_max,torch.abs(self.neg_min))
        self.eps = 0
        self.t = 0
        self.init_state = 0
        self.init_batch = 20

    def __repr__(self):
        return f"ST_BIFNeuron_SS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.acc_q = 0.0
        self.t = 0
        self.init_state = 0

    def forward(self, input):
        # print("input.mean()",input.abs().mean().item(),"self.q_threshold",self.q_threshold.data.item())
        
        # s_grad_scale = 1.0 / (((input).detach().abs().mean() * input.numel()) ** 0.5)
        # if self.init:
        #     if self.init_state == 0:
        #         self.q_threshold.data = (((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
        #         self.init_state += 1
        #     elif self.init_state < self.init_batch:
        #         self.q_threshold.data = 0.1*self.q_threshold.data + 0.9*(((input).detach().abs().mean() * 2) / (self.pos_max.detach().abs() ** 0.5))
        #         self.init_state += 1
        #     else:
        #         self.init = False

        if not torch.is_tensor(self.q) and not torch.is_tensor(self.acc_q):
            self.q = input * 0.0 + 0.5*self.q_threshold
            # self.q = input * 0.0
            self.acc_q = input * 0.0

        # if self.steps > 0:
        #     self.q = self.q + 0.5*self.q_threshold/self.steps
        #     self.steps = self.steps - 1
        
        # s_scale = grad_scale(self.q_threshold, s_grad_scale)
        self.t = self.t + 1
        spikes, self.q, self.acc_q = ST_BIFNodeATGF_SS.apply(input, self.q, self.acc_q, self.q_threshold, self.pos_max, self.neg_min, torch.tensor(self.t))
        return spikes*self.q_threshold


class ST_BIFNodeATGF_MS(torch.autograd.Function):
    """Multi-step ST-BIF Node Autograd Function"""
    
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):

        Time = x_seq.shape[0]
        v_seq = []
        T_seq = []
        H_seq = []
        spike_seq = []
        v = x_seq[0]*0 + 0.5*v_th + prefire*v_th
        T = x_seq[0]*0
        spike = x_seq[0]*0
        T_seq.append(T)
        spike_seq.append(spike)
        H_seq.append(v)
        
        for t in range(Time):
            spike = spike * 0.0
            v = v + x_seq[t]
            H_seq.append(v)
            spike[torch.logical_and((torch.ge(v-v_th,0)), (torch.lt(T-T_max,0)))] = 1
            spike[torch.logical_and((torch.lt(v,0)), (torch.gt(T-T_min,0)))] = -1
            if t < T_max:
                v = v - v_th * spike - prefire*v_th/T_max
            else:
                v = v - v_th * spike
            T = T + spike
            T_seq.append(T)
            spike_seq.append(spike)

        H_seq = torch.stack(H_seq,dim=0)
        T_seq = torch.stack(T_seq,dim=0)
        spike_seq = torch.stack(spike_seq,dim=0)
        
        ctx.save_for_backward(spike_seq,T_seq,H_seq,v_th,T_max,T_min)
        
        return spike_seq[1:,], v, T_seq[1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):

        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        Time = spike_seq.shape[0] - 1
        grad_x_seq = []

        grad_Y_seq = grad_spike_seq
        
        grad_V = 0.0 # t
        grad_T = 0.0 # t
        for t in range(Time, 0, -1):
            grad_T_t_to_H_t = (theta_backward(H_seq[t] - v_th)*theta(T_max - T_seq[t-1])+theta_backward(-H_seq[t])*theta(T_seq[t-1] - T_min))
            grad_Y_t_to_T_t_1 = -(theta_eq(H_seq[t]-v_th)*theta_backward(T_max - T_seq[t-1])+theta(-H_seq[t])*theta_backward(T_seq[t-1] - T_min))
            
            grad_X = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_T_t_to_H_t + grad_V
            grad_T = (grad_Y_seq[t-1] - v_th*grad_V + grad_T)*grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X + 0.0
            grad_x_seq.append(grad_X)
        
        grad_x_seq = torch.flip(torch.stack(grad_x_seq,dim=0),dims=[0])
        return grad_x_seq, None, None, None, None


class ST_BIFNeuron_MS(nn.Module):
    """Multi-step ST-BIF Neuron with CUDA acceleration"""
    
    def __init__(self, q_threshold, level, sym=False, first_neuron=False, need_spike_tracer=False):
        super(ST_BIFNeuron_MS, self).__init__()
        # self.q = 0.0
        self.need_spike_tracer = need_spike_tracer
        if self.need_spike_tracer:
            self.acc_q = 0.0
        self.T = 0
        self.first_neuron = first_neuron
        self.suppress_over_fire = False
        self.overfireLoss = 0.0
        self.name = ""

        # if self.first_neuron:
        #     self.dim = 14
        #     self.time_allocator = nn.Parameter(torch.ones(self.T - 1, 1, 1, 1, 1),requires_grad=True)
        # else:
        #     self.dim = 197
        #     self.time_allocator = nn.Parameter(torch.ones(self.T - 1, 1, 1, 1),requires_grad=True)

        self.q_threshold = nn.Parameter(torch.tensor(q_threshold),requires_grad=False)
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.register_buffer("pos_max",torch.tensor(level//2 - 1))
            self.register_buffer("neg_min",torch.tensor(-level//2 - 1))
            # self.pos_max = torch.tensor(level//2 - 1)
            # self.neg_min = torch.tensor(-level//2)
        else:
            self.register_buffer("pos_max",torch.tensor(level - 1))
            self.register_buffer("neg_min",torch.tensor(0))
            # self.pos_max = torch.tensor(level - 1)
            # self.neg_min = torch.tensor(0)
        self.register_buffer("prefire",torch.tensor(0.0))
        self.init = True
        self.eps = 0
        self.spike_count = 0

    # def __repr__(self):
    #         return f"ST_BIFNeuron_MS(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        # self.q = 0.0
        if self.need_spike_tracer:
            self.acc_q = 0.0

    def forward(self, input):
        N = input.shape[0]
        ori_shape = input.shape

        input = input.reshape(torch.Size([int((self.T)),N//int((self.T))]) + input.shape[1:])
        # print("ST_BIFNeuron_MS input.sum(dim=0).abs().mean()",input.sum(dim=0).abs().mean(),input.dtype)
        
        # s_scale = grad_scale(self.q_threshold, s_grad_scale)
        # print("self.q_threshold",self.q_threshold.item())
        spike_seq, v, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(input.flatten(2), self.q_threshold, self.pos_max, self.neg_min, self.prefire)
        # self.q = v
        # print(self.q[self.q>0].mean())
        
        # calc spike rate over time
        # spike rate = spike count / input size = "non zero" in spike_seq
        
        if self.need_spike_tracer:
            self.acc_q = T_seq.reshape(ori_shape)
        # print("ST_BIFNeuron_MS output.sum(dim=0).abs().mean()",(spike_seq*self.q_threshold).sum(dim=0).abs().mean(),spike_seq.dtype)
        if self.suppress_over_fire:
            self.overfireLoss = ((spike_seq.abs().sum(dim=0) - spike_seq.sum(dim=0).abs())).sum() / spike_seq.numel()
        
        self.spike_count = spike_seq.abs().sum(dim=0).sum()
        
        return spike_seq.reshape(ori_shape)*self.q_threshold


class _ST_BIFNeuron_MS(nn.Module):
    """Legacy ST_BIF Neuron MS implementation (for compatibility)"""
    
    def __init__(self, q_threshold, level, sym=False, first_neuron=False, need_spike_tracer=False):
        super(_ST_BIFNeuron_MS, self).__init__()
        
        self.q = None     
        self.acc_q = None
        
        self.need_spike_tracer = need_spike_tracer
        self.T = 0
        self.first_neuron = first_neuron
        self.suppress_over_fire = False
        self.overfireLoss = 0.0
        self.name = ""