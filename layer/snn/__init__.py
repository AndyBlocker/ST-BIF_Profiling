"""
Spiking Neural Network Framework

A comprehensive framework for spiking neural networks with quantization support.
This module maintains backward compatibility with the original spike_quan_layer.py
and spike_quan_wrapper_ICML.py implementations.

Main components:
- neurons: Various spiking neuron models (IF, ST-BIF)
- layers: Quantized and spiking layer implementations  
- utils: Utility functions and helper classes
- conversion: Model conversion utilities (ANN->QANN->SNN)
- wrapper: SNN wrapper classes and preprocessing
"""

# ============================================================================
# BACKWARD COMPATIBILITY IMPORTS
# ============================================================================
# Import all components to maintain compatibility with original files

# Neurons
from .neurons import (
    ORIIFNeuron, IFNeuron,
    ST_BIFNodeATGF_SS, ST_BIFNeuron_SS,
    ST_BIFNodeATGF_MS, ST_BIFNeuron_MS, _ST_BIFNeuron_MS,
    theta, theta_backward, theta_eq
)

# Import all modules - with proper imports
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

# Import framework-level utilities (now at root level)
try:
    from utils import (
        spiking_softmax, Addition, save_module_inout, Attention_no_softmax,
        set_init_false, cal_overfire_loss,
        save_input_for_bin_snn_5dim, save_input_for_bin_snn_4dim, save_fc_input_for_bin_snn
    )
except ImportError:
    # Fallback for backward compatibility
    print("Warning: utils module not found at root level")
    spiking_softmax = Addition = save_module_inout = Attention_no_softmax = None
    set_init_false = cal_overfire_loss = None
    save_input_for_bin_snn_5dim = save_input_for_bin_snn_4dim = save_fc_input_for_bin_snn = None

# Import framework-level components (now at root level)
try:
    from conversion import myquan_replace_resnet
except ImportError:
    # Fallback for backward compatibility
    print("Warning: conversion module not found at root level")
    myquan_replace_resnet = None

try:
    from wrapper import SNNWrapper_MS
except ImportError:
    # Fallback for backward compatibility  
    print("Warning: wrapper module not found at root level")
    SNNWrapper_MS = None

# ============================================================================
# EXPORT ALL SYMBOLS FOR BACKWARD COMPATIBILITY
# ============================================================================

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
    
    # Utils
    'spiking_softmax', 'Addition', 'save_module_inout', 'Attention_no_softmax',
    'set_init_false', 'cal_overfire_loss',
    'save_input_for_bin_snn_5dim', 'save_input_for_bin_snn_4dim', 'save_fc_input_for_bin_snn',
    
    # Legacy attention (deprecated)
    'QAttention', 'SAttention', 'QWindowAttention', 'SWindowAttention',
]

# ============================================================================
# VERSION AND METADATA
# ============================================================================

__version__ = "0.1.0"
__author__ = "SNN Research Team"
__description__ = "Modular Spiking Neural Network Framework with Quantization Support"

# ============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# ============================================================================

def get_usage_examples():
    """
    Returns usage examples for the SNN framework
    """
    examples = '''
    # Basic neuron usage
    from layer.snn import ST_BIFNeuron_MS, MyQuan
    
    neuron = ST_BIFNeuron_MS(q_threshold=1.0, level=8)
    quantizer = MyQuan(level=8, sym=False)
    
    # Model conversion  
    from conversion import myquan_replace_resnet
    from wrapper import SNNWrapper_MS
    
    # Convert ANN to QANN
    myquan_replace_resnet(model, level=8, weight_bit=32)
    
    # Convert QANN to SNN  
    snn_model = SNNWrapper_MS(ann_model=qann_model, time_step=8, level=8)
    '''
    return examples

def check_compatibility():
    """
    Check if all original components are available for backward compatibility
    """
    missing_components = []
    
    # Check critical components
    critical_components = [
        'ST_BIFNeuron_MS', 'MyQuan', 'LLConv2d_MS', 'LLLinear_MS'
    ]
    
    for component in critical_components:
        if component not in globals():
            missing_components.append(component)
    
    if missing_components:
        print(f"Warning: Missing components for backward compatibility: {missing_components}")
        return False
    else:
        print("✓ All critical components available for backward compatibility")
        return True

# Run compatibility check on import
if __name__ != '__main__':
    try:
        check_compatibility()
    except Exception as e:
        print(f"Compatibility check failed: {e}")

# ============================================================================
# LEGACY COMPATIBILITY LAYER
# ============================================================================

# This ensures that imports like "from spike_quan_layer import MyQuan" 
# continue to work after refactoring
import sys
sys.modules['spike_quan_layer'] = sys.modules[__name__]