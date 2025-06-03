"""
Model Conversion Utilities for Spiking Neural Networks

This package contains utilities for converting between different neural network types:
ANN -> QANN -> SNN pipeline transformations and related conversion functions.
"""

from .quantization import myquan_replace_resnet

# Export all conversion functions
__all__ = [
    'myquan_replace_resnet'
]