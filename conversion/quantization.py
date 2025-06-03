"""
Model Quantization Conversion Functions

This module contains functions for converting neural networks from
standard precision to quantized precision (ANN -> QANN).
"""

import torch.nn as nn
from layer.snn.layers.quantization import MyQuan, QuanConv2d, QuanLinear


def myquan_replace_resnet(model, level, weight_bit=32, is_softmax=True):
    """
    Replace ReLU layers with quantization layers for ResNet models
    
    Args:
        model: Neural network model to modify
        level: Quantization level
        weight_bit: Bit width for weight quantization (default: 32)
        is_softmax: Whether to use softmax (default: True)
    
    Returns:
        Modified model with quantization layers
    """
    index = 0
    cur_index = 0
    
    def get_index(model):
        """Count QAttention layers in the model"""
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            # Check for QAttention (legacy attention mechanism)
            if hasattr(child, '__class__') and 'QAttention' in str(child.__class__):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model, level):
        """Replace ReLU layers with MyQuan quantization layers"""
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.ReLU):
                print(f"level {level}")
                model._modules[name] = MyQuan(level, sym=False)
                is_need = True
            if not is_need:
                _myquan_replace(child, level)
    
    def _weight_quantization(model, weight_bit):
        """Apply weight quantization to Conv2d and Linear layers"""
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, nn.Conv2d):
                model._modules[name] = QuanConv2d(m=child, quan_w_fn=MyQuan(level=2**weight_bit, sym=True))
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = QuanLinear(m=child, quan_w_fn=MyQuan(level=2**weight_bit, sym=True))
                is_need = True
            if not is_need:
                _weight_quantization(child, weight_bit)
                
    # Execute the conversion pipeline
    get_index(model)
    _myquan_replace(model, level)
    if weight_bit < 32:
        _weight_quantization(model, weight_bit)
    
    return model