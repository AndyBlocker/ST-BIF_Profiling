"""
ST-BIF Spiking Neural Network Framework

A comprehensive framework for spiking neural networks with ST-BIF neurons
and quantization support.

Main components:
- neurons: ST-BIF and IF neuron implementations
- layers: Quantized and spiking layer implementations  
- conversion: Model conversion utilities (ANN->QANN->SNN)
"""

# Import neurons
from .neurons import (
    ORIIFNeuron, IFNeuron,
    ST_BIFNodeATGF_SS, ST_BIFNeuron_SS,
    ST_BIFNodeATGF_MS, ST_BIFNeuron_MS, _ST_BIFNeuron_MS,
    theta, theta_backward, theta_eq
)

# Import layers
from .layers import (
    MyQuan, QuanConv2d, QuanLinear,
    grad_scale, floor_pass, round_pass, clip, threshold_optimization,
    LLConv2d, LLConv2d_MS, LLLinear, LLLinear_MS,
    spiking_BatchNorm2d_MS, spiking_BatchNorm2d, Spiking_LayerNorm,
    SpikingBatchNorm2d_MS, MyBatchNorm1d, LN2BNorm, MLP_BN, 
    MyBachNorm, MyLayerNorm,
    SpikeMaxPooling, SpikeMaxPooling_MS,
    QAttention, SAttention, QWindowAttention, SWindowAttention
)

# Import conversion functions
from .conversion import myquan_replace_resnet

# Export all symbols for backward compatibility
__all__ = [
    # Neurons
    'ORIIFNeuron', 'IFNeuron',
    'ST_BIFNodeATGF_SS', 'ST_BIFNeuron_SS', 
    'ST_BIFNodeATGF_MS', 'ST_BIFNeuron_MS', '_ST_BIFNeuron_MS',
    'theta', 'theta_backward', 'theta_eq',
    
    # Quantization
    'MyQuan', 'QuanConv2d', 'QuanLinear',
    'grad_scale', 'floor_pass', 'round_pass', 'clip', 'threshold_optimization',
    
    # Layers
    'LLConv2d', 'LLConv2d_MS', 'LLLinear', 'LLLinear_MS',
    
    # Normalization  
    'spiking_BatchNorm2d_MS', 'spiking_BatchNorm2d', 'Spiking_LayerNorm',
    'SpikingBatchNorm2d_MS', 'MyBatchNorm1d', 'LN2BNorm', 'MLP_BN', 
    'MyBachNorm', 'MyLayerNorm',
    
    # Pooling
    'SpikeMaxPooling', 'SpikeMaxPooling_MS',
    
    # Legacy attention (deprecated)
    'QAttention', 'SAttention', 'QWindowAttention', 'SWindowAttention',
    
    # Conversion
    'myquan_replace_resnet',
]

__version__ = "0.1.0"
__author__ = "ST-BIF Research Team"
__description__ = "ST-BIF Spiking Neural Network Framework"