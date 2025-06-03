"""
Model Reset Utilities for Spiking Neural Networks

This module contains functions for resetting neuron states 
in SNN models between inference steps.
"""

from snn.neurons import ST_BIFNeuron_MS, IFNeuron


def reset_model(wrapper):
    """Reset all neurons in the SNN model"""
    def _reset_neurons(module):
        children = list(module.named_children())
        for name, child in children:
            if hasattr(child, 'reset'):
                child.reset()
            elif isinstance(child, (ST_BIFNeuron_MS, IFNeuron)):
                if hasattr(child, 'v'):
                    child.v = 0
                if hasattr(child, 'mem'):
                    child.mem = 0
            else:
                _reset_neurons(child)
    
    _reset_neurons(wrapper.model)