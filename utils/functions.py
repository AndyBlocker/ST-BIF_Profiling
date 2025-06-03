"""
Utility Functions for Spiking Neural Networks

This module contains utility functions for model analysis, initialization,
and loss calculation in spiking neural networks.
"""

import torch


def set_init_false(model):
    """Set initialization state to false for all MyQuan modules"""
    # Import locally to avoid circular imports
    from ..layers.quantization import MyQuan
    
    def set_init_false_inner(model):
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, MyQuan):
                model._modules[name].init_state = model._modules[name].batch_init
            else:
                set_init_false_inner(child)
    set_init_false_inner(model)


def cal_overfire_loss(model):
    """Calculate overfire loss from all ST_BIFNeuron_MS modules"""
    # Import locally to avoid circular imports
    from ..neurons.st_bif_neurons import ST_BIFNeuron_MS
    
    l2_loss = 0.0
    def l2_regularization_inner(model):
        nonlocal l2_loss
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, ST_BIFNeuron_MS):
                l2_loss = l2_loss + child.overfireLoss
            else:
                l2_regularization_inner(child)
    l2_regularization_inner(model)
    return l2_loss