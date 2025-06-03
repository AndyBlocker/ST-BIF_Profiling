"""
Integrate-and-Fire Neuron Models

This module contains traditional IF neuron implementations including
original IF neurons and standard IF neurons with different configurations.
"""

import torch
import torch.nn as nn


class ORIIFNeuron(nn.Module):
    """Original Integrate-and-Fire Neuron implementation"""
    
    def __init__(self, q_threshold, level, sym=False):
        super(ORIIFNeuron, self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        self.pos_max = torch.tensor(level - 1)
        self.neg_min = torch.tensor(0)
            
        self.eps = 0

    # def __repr__(self):
    #         return f"IFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        # self.neg_spike_position = None

    def forward(self, input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0)
        # neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        # self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        # self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold


class IFNeuron(nn.Module):
    """Standard Integrate-and-Fire Neuron with symmetric/asymmetric spike levels"""
    
    def __init__(self, q_threshold, level, sym=False):
        super(IFNeuron, self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(-level//2)
        else:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_min = torch.tensor(0)
            
        self.eps = 0

    def __repr__(self):
        return f"ST-BIFNeuron(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, q_threshold={self.q_threshold})"
    
    def reset(self):
        # print("IFNeuron reset")
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self, input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x*self.q_threshold
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float32).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float32).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach() if torch.is_tensor(x) else x)
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        # print((x == 0).all(), (self.cur_output==0).all())
        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold