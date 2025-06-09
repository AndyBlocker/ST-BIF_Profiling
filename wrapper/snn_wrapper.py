from networkx import from_nested_tuple
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional
from copy import deepcopy

# Import SNN components
from snn.neurons import ST_BIFNeuron_MS, IFNeuron, ORIIFNeuron
# from snn.neurons import IFNeuron, ORIIFNeuron
# from snn.neurons import ST_BIFNeuron_MS
# from snn.neurons.st_bif_optimized import ST_BIFNeuron_MS_Optimized as ST_BIFNeuron_MS
from snn.layers import (
    MyQuan, LLConv2d_MS, LLLinear_MS, QuanConv2d, QuanLinear,
    SpikingBatchNorm2d_MS, MyBatchNorm1d, Spiking_LayerNorm,
    SpikeMaxPooling_MS, QAttention, SAttention, QWindowAttention, SWindowAttention
)
from utils import save_module_inout
import glo

# Import local wrapper utilities
from .base import Judger
from .encoding import get_subtensors
from .reset import reset_model
from .attention_conversion import attn_convert, attn_convert_Swin

class SNNWrapper_MS(nn.Module):
    """
    Multi-Step Spiking Neural Network Wrapper
    
    Converts ANN/QANN models to SNN with temporal processing over multiple time steps.
    """
    
    def __init__(self, ann_model, cfg=None, time_step=8, Encoding_type="analog", learnable=False, **kwargs):
        super(SNNWrapper_MS, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs.get("level", 8)
        self.step = self.level//2 - 1
        self.neuron_type = kwargs.get("neuron_type", "ST-BIF")
        self.model = ann_model

        self.model.spike = True
        self.model.T = time_step
        self.model.step = self.step

        self.kwargs = kwargs
        self.model_name = kwargs.get("model_name", "resnet")
        self.is_softmax = kwargs.get("is_softmax", False)
        self.record_inout = kwargs.get("record_inout", False)
        self.record_dir = kwargs.get("record_dir", "./output")
        self.suppress_over_fire = kwargs.get("suppress_over_fire", False)
        self.learnable = learnable
        self.max_T = 0
        self.visualize = False
        self.first_neuron = True
        self.blockNum = 0
        
        if self.model_name.count("vit") > 0:
            if hasattr(self.model, 'pos_embed'):
                self.pos_embed = deepcopy(self.model.pos_embed.data)
            if hasattr(self.model, 'cls_token'):
                self.cls_token = deepcopy(self.model.cls_token.data)

        self._replace_weight(self.model)
        
        if self.record_inout:
            self.calOrder = []
            self._record_inout(self.model)
            self.set_snn_save_name(self.model)
            try:
                local_rank = torch.distributed.get_rank()
            except:
                local_rank = 0
            glo._init()
            if local_rank == 0:
                if not os.path.exists(self.record_dir):
                    os.mkdir(self.record_dir)
                glo.set_value("output_bin_snn_dir", self.record_dir)
                f = open(f"{self.record_dir}/calculationOrder.txt", "w+")
                for order in self.calOrder:
                    f.write(order+"\n")
                f.close()
        
        # Debug information
        print("=====================")
        # print("self.model", self.model)
        print("self.model_name", self.model_name)
        print("self.kwargs", self.kwargs)
        print("self.T:", self.T)
        print("self.level:", self.level)
        print("self.step:", self.step)
        
    def hook_mid_feature(self):
        """Hook for capturing intermediate features"""
        self.feature_list = []
        self.input_feature_list = []
        def _hook_mid_feature(module, input, output):
            self.feature_list.append(output)
            self.input_feature_list.append(input[0])
        
        # This assumes specific model structure - adapt as needed
        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 11:
            self.model.blocks[11].norm2[1].register_forward_hook(_hook_mid_feature)
    
    def get_mid_feature(self):
        """Get captured intermediate features"""
        if hasattr(self, 'feature_list'):
            self.feature_list = torch.stack(self.feature_list, dim=0)
            self.input_feature_list = torch.stack(self.input_feature_list, dim=0)
            print("self.feature_list", self.feature_list.shape) 
            print("self.input_feature_list", self.input_feature_list.shape)
            
    def reset(self):
        """Reset the SNN model state"""
        if self.model_name.count("vit") > 0:
            if hasattr(self.model, 'pos_embed') and hasattr(self, 'pos_embed'):
                self.model.pos_embed.data = deepcopy(self.pos_embed).cuda()
            if hasattr(self.model, 'cls_token') and hasattr(self, 'cls_token'):
                self.model.cls_token.data = deepcopy(self.cls_token).cuda()
        
        reset_model(self)

    def _record_inout(self, model):
        """Setup input/output recording for debugging"""
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, SAttention):
                model._modules[name].first = True
                model._modules[name].T = self.T
                is_need = True
            elif isinstance(child, nn.Sequential) and len(child) > 1 and isinstance(child[1], IFNeuron):
                model._modules[name] = save_module_inout(m=child, T=self.T)
                model._modules[name].first = True
                is_need = True
            if not is_need:            
                self._record_inout(child)            

    def set_snn_save_name(self, model):
        """Set names for saved modules during recording"""
        children = list(model.named_modules())
        for name, child in children:
            if isinstance(child, save_module_inout):
                child.name = name
                self.calOrder.append(name)
            if isinstance(child, SAttention):
                child.name = name
                self.calOrder.append(name)
    
    def _replace_weight(self, model):
        """Replace ANN/QANN layers with SNN equivalents"""
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            
            if isinstance(child, QAttention):
                SAttn = SAttention(
                    dim=child.num_heads*child.head_dim,
                    num_heads=child.num_heads,
                    level=self.level,
                    is_softmax=self.is_softmax,
                    neuron_layer=ST_BIFNeuron_MS,
                    T=self.T
                )
                attn_convert(QAttn=child, SAttn=SAttn, level=self.level, 
                           neuron_type=self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
                
            elif isinstance(child, QWindowAttention):
                self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention(
                    dim=child.num_heads*child.head_dim, 
                    window_size=child.window_size,
                    num_heads=child.num_heads,
                    level=self.level,
                    neuron_layer=ST_BIFNeuron_MS,
                    T=self.T
                )
                attn_convert_Swin(QAttn=child, SAttn=SAttn, level=self.level,
                                neuron_type=self.neuron_type, T=self.T, 
                                suppress_over_fire=self.suppress_over_fire)
                model._modules[name] = SAttn
                is_need = True
                
            elif isinstance(child, (nn.Conv2d, QuanConv2d)):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/(self.level//2 - 1)
                    print(f"{model._modules[name]}'s bias divided by {self.level//2 - 1}")
                model._modules[name] = LLConv2d_MS(child, time_step=self.T, **self.kwargs)
                is_need = True
                
            elif isinstance(child, (nn.Linear, QuanLinear)):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/(self.level//2 - 1)
                model._modules[name] = LLLinear_MS(child, time_step=self.T, **self.kwargs)
                is_need = True
                
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling_MS(child, time_step=self.T)
                is_need = True
                
            elif isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, MyBatchNorm1d)):
                model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name].running_mean = model._modules[name].running_mean/self.step
                model._modules[name].spike = True
                model._modules[name].T = self.T
                model._modules[name].step = self.step
                model._modules[name] = SpikingBatchNorm2d_MS(bn=child, level=self.level, time_step=self.T)
                is_need = True
                
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0], T=self.T)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
                
            elif isinstance(child, MyQuan):
                neurons = ST_BIFNeuron_MS(
                    q_threshold=torch.tensor(1.0),
                    sym=child.sym,
                    level=child.pos_max, 
                    first_neuron=self.first_neuron
                )
                neurons.q_threshold.data = child.s.data
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.init = True
                neurons.T = self.T
                self.first_neuron = False
                neurons.cuda()
                model._modules[name] = neurons
                is_need = True
                
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
                
            if not is_need:            
                self._replace_weight(child)

    def _reset_all_states(self):
        """Reset all neuron and layer states in the model"""
        def _reset_module(module):
            # Reset specific layer types
            if hasattr(module, 'reset'):
                module.reset()
            # Handle specific neuron types
            elif hasattr(module, 'v'):  # Neuron with membrane potential
                if isinstance(module.v, (int, float)):
                    module.v = 0
                elif hasattr(module.v, 'zero_'):
                    module.v.zero_()
            elif hasattr(module, 'mem'):  # Neuron with memory
                if isinstance(module.mem, (int, float)):
                    module.mem = 0
                elif hasattr(module.mem, 'zero_'):
                    module.mem.zero_()
            
            # Recursively reset children
            for child in module.children():
                _reset_module(child)
        
        _reset_module(self.model)

    def forward(self, x, verbose=False):
        """
        Forward pass through the SNN
        
        Args:
            x: Input tensor
            verbose: Whether to return detailed temporal information
            
        Returns:
            SNN output (accumulated over time steps)
        """
        # Reset all neuron and layer states before each forward pass
        self._reset_all_states()
        
        input = get_subtensors(x, 0.0, 0.0, sample_grain=self.step, time_step=self.T)  
        
        T, B, C, H, W = input.shape
        input = input.reshape(T*B, C, H, W)
        output = self.model(input)
        output = output.reshape(torch.Size([T, B]) + output.shape[1:])
        
        if verbose:
            accu_per_t = []
            accu = 0.0
            for t in range(T):
                accu = accu + output[t]
                accu_per_t.append(accu + 0.0)
            return output.sum(dim=0), self.T, torch.stack(accu_per_t, dim=0)
        
        # Count spikes for analysis
        spike_count = 0
        def check_spike_count(child):
            nonlocal spike_count
            children = list(child.named_children())
            for name, module in children:
                if isinstance(module, ST_BIFNeuron_MS):
                    if hasattr(module, 'spike_count'):
                        spike_count += module.spike_count
                else: 
                    check_spike_count(module)
        
        check_spike_count(self.model)
        self.reset()
        return output.sum(dim=0)
    
    def _replace_weight(self, model):
        """Replace ANN/QANN layers with SNN equivalents"""
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            
            if isinstance(child, QAttention):
                SAttn = SAttention(
                    dim=child.num_heads*child.head_dim,
                    num_heads=child.num_heads,
                    level=self.level,
                    is_softmax=self.is_softmax,
                    neuron_layer=ST_BIFNeuron_MS,
                    T=self.T
                )
                attn_convert(QAttn=child, SAttn=SAttn, level=self.level, 
                           neuron_type=self.neuron_type, T=self.T)
                model._modules[name] = SAttn
                is_need = True
                
            elif isinstance(child, QWindowAttention):
                self.blockNum = self.blockNum + 1/24
                SAttn = SWindowAttention(
                    dim=child.num_heads*child.head_dim, 
                    window_size=child.window_size,
                    num_heads=child.num_heads,
                    level=self.level,
                    neuron_layer=ST_BIFNeuron_MS,
                    T=self.T
                )
                attn_convert_Swin(QAttn=child, SAttn=SAttn, level=self.level,
                                neuron_type=self.neuron_type, T=self.T, 
                                suppress_over_fire=self.suppress_over_fire)
                model._modules[name] = SAttn
                is_need = True
                
            elif isinstance(child, (nn.Conv2d, QuanConv2d)):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/(self.level//2 - 1)
                    print(f"{model._modules[name]}'s bias divided by {self.level//2 - 1}")
                model._modules[name] = LLConv2d_MS(child, time_step=self.T, **self.kwargs)
                is_need = True
                
            elif isinstance(child, (nn.Linear, QuanLinear)):
                if child.bias is not None:
                    model._modules[name].bias.data = model._modules[name].bias.data/(self.level//2 - 1)
                model._modules[name] = LLLinear_MS(child, time_step=self.T, **self.kwargs)
                is_need = True
                
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling_MS(child, time_step=self.T)
                is_need = True
                
            elif isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, MyBatchNorm1d)):
                model._modules[name].bias.data = model._modules[name].bias.data/self.step
                model._modules[name].running_mean = model._modules[name].running_mean/self.step
                model._modules[name].spike = True
                model._modules[name].T = self.T
                model._modules[name].step = self.step
                model._modules[name] = SpikingBatchNorm2d_MS(bn=child, level=self.level, time_step=self.T)
                is_need = True
                
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.normalized_shape[0], T=self.T)
                SNN_LN.layernorm = child
                if child.elementwise_affine:
                    SNN_LN.weight = child.weight.data
                    SNN_LN.bias = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
                
            elif isinstance(child, MyQuan):
                neurons = ST_BIFNeuron_MS(
                    q_threshold=torch.tensor(1.0),
                    sym=child.sym,
                    level=child.pos_max, 
                    first_neuron=self.first_neuron
                )
                neurons.q_threshold.data = child.s.data
                neurons.level = self.level
                neurons.pos_max = child.pos_max
                neurons.neg_min = child.neg_min
                neurons.init = True
                neurons.T = self.T
                self.first_neuron = False
                neurons.cuda()
                model._modules[name] = neurons
                is_need = True
                
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
                
            if not is_need:            
                self._replace_weight(child)