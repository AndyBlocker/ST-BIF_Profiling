"""
Temporal Encoding Utilities for Spiking Neural Networks

This module contains functions for converting analog inputs to 
temporal spike sequences for SNN processing.
"""

import torch


def get_subtensors(tensor, mean, std, sample_grain=255, time_step=4):
    """
    Convert analog tensor to temporal spike sequences (original implementation)
    
    Args:
        tensor: Input analog tensor
        mean: Mean value (unused in current implementation)
        std: Standard deviation (unused in current implementation)  
        sample_grain: Quantization grain, default 255
        time_step: Number of time steps, default 4
    
    Returns:
        Temporal spike tensor of shape [T, *original_tensor_shape]
    """
    for i in range(int(time_step)):
        # Normalize tensor by sample_grain
        output = (tensor/sample_grain).unsqueeze(0)
        if i == 0:
            accu = output
        elif i < sample_grain:
            accu = torch.cat((accu, output), dim=0)
        else:
            accu = torch.cat((accu, output*0.0), dim=0)
    
    return accu