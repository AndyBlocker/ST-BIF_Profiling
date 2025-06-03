"""
Spiking Neural Network Layers

This package contains layer implementations for spiking neural networks,
organized by functionality: quantization, convolution, linear, normalization, pooling.
"""

# Import all layer types
from .quantization import (
    MyQuan, QuanConv2d, QuanLinear,
    grad_scale, floor_pass, round_pass, clip, threshold_optimization
)

from .conv import LLConv2d, LLConv2d_MS

from .linear import LLLinear, LLLinear_MS

from .normalization import (
    spiking_BatchNorm2d_MS, spiking_BatchNorm2d, Spiking_LayerNorm,
    SpikingBatchNorm2d_MS, MyBatchNorm1d, LN2BNorm, MLP_BN, 
    MyBachNorm, MyLayerNorm
)

from .pooling import SpikeMaxPooling, SpikeMaxPooling_MS

# Legacy attention (marked for deprecation)
# Note: These are placeholders since attention modules are being phased out
try:
    # Placeholder attention classes for backward compatibility
    class QAttention:
        def __init__(self, *args, **kwargs):
            # Import original for compatibility
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from spike_quan_layer import QAttention as OriginalClass
            self._original = OriginalClass(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._original, name)
    
    class SAttention:
        def __init__(self, *args, **kwargs):
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from spike_quan_layer import SAttention as OriginalClass
            self._original = OriginalClass(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._original, name)
    
    class QWindowAttention:
        def __init__(self, *args, **kwargs):
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from spike_quan_layer import QWindowAttention as OriginalClass
            self._original = OriginalClass(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._original, name)
    
    class SWindowAttention:
        def __init__(self, *args, **kwargs):
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from spike_quan_layer import SWindowAttention as OriginalClass
            self._original = OriginalClass(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._original, name)
            
except ImportError:
    pass  # Attention modules are being phased out

__all__ = [
    # Quantization
    'MyQuan', 'QuanConv2d', 'QuanLinear',
    'grad_scale', 'floor_pass', 'round_pass', 'clip', 'threshold_optimization',
    
    # Convolution
    'LLConv2d', 'LLConv2d_MS',
    
    # Linear  
    'LLLinear', 'LLLinear_MS',
    
    # Normalization
    'spiking_BatchNorm2d_MS', 'spiking_BatchNorm2d', 'Spiking_LayerNorm',
    'SpikingBatchNorm2d_MS', 'MyBatchNorm1d', 'LN2BNorm', 'MLP_BN', 
    'MyBachNorm', 'MyLayerNorm',
    
    # Pooling
    'SpikeMaxPooling', 'SpikeMaxPooling_MS',
    
    # Legacy attention (deprecated)
    'QAttention', 'SAttention', 'QWindowAttention', 'SWindowAttention'
]