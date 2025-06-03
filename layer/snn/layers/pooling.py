"""
Spiking Neural Network Pooling Layers

This module contains pooling layer implementations for spiking neural networks,
including spiking max pooling with temporal accumulation for both single-step
and multi-step processing modes.
"""

import torch
import torch.nn as nn


class SpikeMaxPooling(nn.Module):
    """Single-step spiking max pooling with temporal accumulation"""
    
    def __init__(self, maxpool):
        super(SpikeMaxPooling, self).__init__()
        self.maxpool = maxpool
        
        self.accumulation = None
    
    def reset(self):
        self.accumulation = None

    def forward(self, x):
        old_accu = self.accumulation
        if self.accumulation is None:
            self.accumulation = x
        else:
            self.accumulation = self.accumulation + x
        
        if old_accu is None:
            output = self.maxpool(self.accumulation)
        else:
            output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

        # print("output.shape",output.shape)
        # print(output[0][0][0:4][0:4])
        
        return output


class SpikeMaxPooling_MS(nn.Module):
    """Multi-step spiking max pooling with temporal accumulation"""
    
    def __init__(self, maxpool, **kwargs):
        super(SpikeMaxPooling_MS, self).__init__()
        self.maxpool = maxpool
        self.T = kwargs.get("time_step", 1)  # 总时间步长
        self.reset()
        
    def reset(self):
        """重置所有状态"""
        self.prev_pooled_accum = None
        self.accumulation = None
        
    def forward(self, x):
        # 每次前向传递开始时重置状态，避免不同批次大小的问题
        self.reset()
        
        total_batch = x.shape[0]
        B = total_batch // self.T
        device = x.device
        
        outputs = []
        
        # 按时间步处理
        for t in range(self.T):
            # 获取当前时间步的所有样本
            t_start = t * B
            t_end = (t + 1) * B
            current_inputs = x[t_start:t_end]
            
            # 更新累积和
            if self.accumulation is None:
                self.accumulation = current_inputs
            else:
                # 确保维度匹配
                if self.accumulation.shape[0] != current_inputs.shape[0]:
                    # 如果批次大小改变，重置累积状态
                    self.accumulation = current_inputs
                    self.prev_pooled_accum = None
                else:
                    self.accumulation = self.accumulation + current_inputs
            
            # 计算当前累积和的maxpool结果
            current_pooled = self.maxpool(self.accumulation)
            
            # 计算输出
            if self.prev_pooled_accum is None:
                output = current_pooled
            else:
                output = current_pooled - self.prev_pooled_accum
            
            # 更新前一个时间步的累积池化结果
            self.prev_pooled_accum = current_pooled
            
            outputs.append(output)
        
        # 拼接所有时间步的输出
        return torch.cat(outputs, dim=0) if outputs else x.new_tensor([])