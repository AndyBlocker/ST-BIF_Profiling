"""
Base Classes for SNN Wrappers

This module contains base classes and utilities for SNN wrapper implementations.
"""

import torch.nn as nn


class Judger(nn.Module):
    """Judger class for managing SNN execution flow"""
    def __init__(self):
        super(Judger, self).__init__()
        self.finished = False
        
    def reset(self):
        self.finished = False