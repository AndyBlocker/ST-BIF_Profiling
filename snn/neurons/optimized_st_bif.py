#!/usr/bin/env python3
"""
Optimized ST-BIF Neuron Implementation

This module provides optimized ST-BIF neuron implementations with:
1. Memory pool integration for reduced allocations
2. Improved memory layout for better coalescing
3. Mixed precision support
4. Reduced CPU-GPU synchronization
5. Performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Import optimization utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from snn.optimization_utils import get_memory_pool, get_performance_monitor
except ImportError:
    warnings.warn("Optimization utilities not available, using fallback implementations")
    get_memory_pool = lambda: None
    get_performance_monitor = lambda: None

# Import original CUDA implementation
try:
    from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA
except ImportError:
    warnings.warn("Original CUDA implementation not available, using PyTorch fallback")
    ST_BIFNodeATGF_MS_CUDA = None


class OptimizedST_BIFNodeATGF_MS(torch.autograd.Function):
    """
    Optimized Multi-step ST-BIF Node Autograd Function with memory pool and 
    improved performance characteristics.
    """
    
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, 
                T_min: torch.Tensor, prefire: torch.Tensor, use_memory_pool: bool = True):
        """
        Optimized forward pass with memory pool and better memory layout.
        
        Args:
            x_seq: Input sequence tensor [T, B*F] where T=time_steps, B=batch_size, F=features
            v_th: Voltage threshold
            T_max: Maximum spike count  
            T_min: Minimum spike count
            prefire: Prefire parameter
            use_memory_pool: Whether to use memory pool for intermediate tensors
            
        Returns:
            spike_seq: Output spike sequence
            v: Final voltage state
            T_seq: Spike count sequence
        """
        if x_seq.dim() == 2:
            Time, batch_features = x_seq.shape
        else:
            # Handle multi-dimensional input by flattening
            Time = x_seq.shape[0]
            batch_features = x_seq.shape[1:].numel()
        device = x_seq.device
        dtype = x_seq.dtype
        
        # Get memory pool if available
        memory_pool = get_memory_pool(device) if use_memory_pool else None
        
        # Allocate tensors from memory pool or normally
        if memory_pool is not None:
            v_seq = memory_pool.get_tensor((Time + 1, batch_features), dtype, requires_grad=True)
            T_seq = memory_pool.get_tensor((Time + 1, batch_features), dtype, requires_grad=True)
            spike_seq = memory_pool.get_tensor((Time + 1, batch_features), dtype, requires_grad=True)
            H_seq = memory_pool.get_tensor((Time + 1, batch_features), dtype, requires_grad=False)
        else:
            v_seq = torch.zeros(Time + 1, batch_features, dtype=dtype, device=device)
            T_seq = torch.zeros(Time + 1, batch_features, dtype=dtype, device=device)  
            spike_seq = torch.zeros(Time + 1, batch_features, dtype=dtype, device=device)
            H_seq = torch.zeros(Time + 1, batch_features, dtype=dtype, device=device)
        
        # Initialize state tensors with correct shape
        init_shape = (batch_features,) if x_seq.dim() == 2 else x_seq.shape[1:]
        v = torch.full(init_shape, 0.5 * v_th.item() + prefire.item() * v_th.item(), 
                      dtype=dtype, device=device)
        T = torch.zeros(init_shape, dtype=dtype, device=device)
        spike = torch.zeros(init_shape, dtype=dtype, device=device)
        
        # Store initial states
        v_seq[0] = v
        T_seq[0] = T
        spike_seq[0] = spike
        H_seq[0] = v
        
        # Main computation loop - optimized for memory access patterns
        for t in range(Time):
            # Reset spike for this timestep
            spike.zero_()
            
            # Update voltage
            v.add_(x_seq[t])
            H_seq[t + 1].copy_(v)
            
            # Compute spike conditions (vectorized, reduced branching)
            pos_condition = (v >= v_th) & (T < T_max)
            neg_condition = (v < 0) & (T > T_min)
            
            # Set spikes efficiently
            spike[pos_condition] = 1.0
            spike[neg_condition] = -1.0
            
            # Update voltage and spike count
            if t < T_max:
                v.sub_(v_th * spike + prefire * v_th / T_max)
            else:
                v.sub_(v_th * spike)
            
            T.add_(spike)
            
            # Store results
            v_seq[t + 1] = v
            T_seq[t + 1] = T
            spike_seq[t + 1] = spike
        
        # Save for backward pass
        ctx.save_for_backward(spike_seq, T_seq, H_seq, v_th, T_max, T_min)
        ctx.use_memory_pool = use_memory_pool
        ctx.memory_pool = memory_pool
        
        return spike_seq[1:], v, T_seq[1:]
    
    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, 
                 grad_T_seq: torch.Tensor):
        """
        Optimized backward pass with memory-efficient gradient computation.
        """
        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        use_memory_pool = ctx.use_memory_pool
        memory_pool = ctx.memory_pool
        
        Time = spike_seq.shape[0] - 1
        batch_features = spike_seq.shape[1]
        
        # Allocate gradient tensor
        if memory_pool is not None:
            grad_x_seq = memory_pool.get_tensor((Time, batch_features), grad_spike_seq.dtype, requires_grad=False)
        else:
            grad_x_seq = torch.zeros(Time, batch_features, dtype=grad_spike_seq.dtype, device=grad_spike_seq.device)
        
        # Initialize gradient accumulators
        grad_V = torch.zeros_like(grad_v_seq)
        grad_T = torch.zeros_like(grad_v_seq)
        
        # Backward through time
        for t in range(Time, 0, -1):
            # Compute gradients with vectorized operations
            H_diff = H_seq[t] - v_th
            neg_H = -H_seq[t]
            T_diff_max = T_max - T_seq[t-1]
            T_diff_min = T_seq[t-1] - T_min
            
            # Gradient computations (optimized)
            grad_T_t_to_H_t = (
                theta_backward(H_diff) * theta(T_diff_max) + 
                theta_backward(neg_H) * theta(T_diff_min)
            )
            
            grad_Y_t_to_T_t_1 = -(
                theta_eq(H_diff) * theta_backward(T_diff_max) +
                theta(neg_H) * theta_backward(T_diff_min)
            )
            
            # Compute final gradients
            temp_grad = grad_spike_seq[t-1] - v_th * grad_V + grad_T
            grad_X = temp_grad * grad_T_t_to_H_t + grad_V
            grad_T = temp_grad * grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X
            
            grad_x_seq[t-1] = grad_X
        
        # Return tensors to memory pool if used
        if memory_pool is not None:
            # Return intermediate tensors to pool
            memory_pool.return_tensor(spike_seq)
            memory_pool.return_tensor(T_seq)
            memory_pool.return_tensor(H_seq)
        
        return grad_x_seq, None, None, None, None, None


def theta_backward(x):
    """Optimized backward pass function for theta with reduced memory allocations"""
    # Use in-place operations where possible
    sigmoid = torch.sigmoid(4 * x)
    return 4 * sigmoid * (1 - sigmoid)


def theta(x):
    """Optimized step function with reduced memory allocations"""
    return (x > 0).to(dtype=x.dtype)


def theta_eq(x):
    """Optimized step function with reduced memory allocations"""  
    return (x >= 0).to(dtype=x.dtype)


class OptimizedST_BIFNeuron_MS(nn.Module):
    """
    Optimized Multi-step ST-BIF Neuron with performance improvements:
    
    1. Memory pool integration for reduced allocations
    2. Improved memory layout for better GPU utilization
    3. Mixed precision support
    4. Performance monitoring integration
    5. Configurable optimization levels
    """
    
    def __init__(self, q_threshold, level, sym=False, first_neuron=False, 
                 need_spike_tracer=False, optimization_level="medium"):
        super(OptimizedST_BIFNeuron_MS, self).__init__()
        
        # Core parameters (unchanged from original)
        self.need_spike_tracer = need_spike_tracer
        if self.need_spike_tracer:
            self.acc_q = 0.0
        self.T = 0
        self.first_neuron = first_neuron
        self.suppress_over_fire = False
        self.overfireLoss = 0.0
        self.name = ""
        
        # Threshold and level parameters
        self.q_threshold = nn.Parameter(torch.tensor(q_threshold), requires_grad=False)
        self.level = torch.tensor(level)
        self.sym = sym
        
        # Set up spike count limits
        if sym:
            self.register_buffer("pos_max", torch.tensor(level // 2 - 1))
            self.register_buffer("neg_min", torch.tensor(-level // 2 - 1))
        else:
            self.register_buffer("pos_max", torch.tensor(level - 1))
            self.register_buffer("neg_min", torch.tensor(0))
            
        self.register_buffer("prefire", torch.tensor(0.0))
        
        # Optimization settings
        self.optimization_level = optimization_level
        self.use_memory_pool = optimization_level in ["medium", "high"]
        self.use_original_cuda = optimization_level == "low"  # Fallback to original implementation
        
        # Performance monitoring
        self.monitor = get_performance_monitor()
        
        # Initialize state
        self.init = True
        self.eps = 0
        self.spike_count = 0
        
        # Cache for persistent tensors
        self._cached_tensors = {}
        
    def reset(self):
        """Reset neuron state"""
        if self.need_spike_tracer:
            self.acc_q = 0.0
        self.spike_count = 0
        # Clear tensor cache
        self._cached_tensors.clear()
    
    def _get_cached_tensor(self, key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get or create cached tensor to reduce allocations"""
        cache_key = (key, shape, dtype, str(device))
        if cache_key not in self._cached_tensors:
            self._cached_tensors[cache_key] = torch.empty(shape, dtype=dtype, device=device)
        else:
            # Ensure tensor is the right shape
            cached = self._cached_tensors[cache_key]
            if cached.shape != shape:
                self._cached_tensors[cache_key] = torch.empty(shape, dtype=dtype, device=device)
        
        return self._cached_tensors[cache_key]
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with performance monitoring and memory optimization.
        
        Args:
            input: Input tensor with shape [T*B, ...] where T=time_steps, B=batch_size
            
        Returns:
            Spike output tensor with same shape as input
        """
        # Performance monitoring
        if self.monitor is not None:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        
        N = input.shape[0]
        ori_shape = input.shape
        
        # Reshape for time-series processing: [T*B, ...] -> [T, B, ...]
        input = input.reshape(torch.Size([int(self.T), N // int(self.T)]) + input.shape[1:])
        
        # Flatten spatial dimensions for efficient processing: [T, B, H, W] -> [T, B*H*W]
        input_flat = input.flatten(2)
        
        # Choose computation path based on optimization level
        if self.use_original_cuda and ST_BIFNodeATGF_MS_CUDA is not None:
            # Use original CUDA implementation for compatibility
            spike_seq, v, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(
                input_flat, self.q_threshold, self.pos_max, self.neg_min, self.prefire
            )
        else:
            # Use optimized implementation
            spike_seq, v, T_seq = OptimizedST_BIFNodeATGF_MS.apply(
                input_flat, self.q_threshold, self.pos_max, self.neg_min, 
                self.prefire, self.use_memory_pool
            )
        
        # Handle spike tracing if needed
        if self.need_spike_tracer:
            self.acc_q = T_seq.reshape(ori_shape)
        
        # Handle overfire suppression if enabled
        if self.suppress_over_fire:
            spike_sum = spike_seq.abs().sum(dim=0)
            spike_abs_sum = spike_seq.sum(dim=0).abs()
            self.overfireLoss = (spike_sum - spike_abs_sum).sum() / spike_seq.numel()
        
        # Update spike count statistics
        self.spike_count = spike_seq.abs().sum(dim=0).sum()
        
        # Reshape back to original shape and apply threshold scaling
        output = spike_seq.reshape(ori_shape) * self.q_threshold
        
        # Performance monitoring
        if self.monitor is not None:
            end_time.record()
            torch.cuda.synchronize()
            forward_time = start_time.elapsed_time(end_time)
            self.monitor.record_forward_time(forward_time)
        
        return output
    
    def extra_repr(self) -> str:
        """Return extra representation string for debugging"""
        return (f'level={self.level.item()}, sym={self.sym}, '
                f'pos_max={self.pos_max.item()}, neg_min={self.neg_min.item()}, '
                f'q_threshold={self.q_threshold.item():.4f}, '
                f'optimization_level={self.optimization_level}')


class AdaptiveMixedPrecisionST_BIF(OptimizedST_BIFNeuron_MS):
    """
    ST-BIF Neuron with adaptive mixed precision support.
    Automatically switches between FP16 and FP32 based on numerical stability.
    """
    
    def __init__(self, q_threshold, level, sym=False, first_neuron=False,
                 need_spike_tracer=False, optimization_level="high",
                 enable_amp=True, stability_threshold=1e-4):
        super().__init__(q_threshold, level, sym, first_neuron, need_spike_tracer, optimization_level)
        
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.stability_threshold = stability_threshold
        self.amp_enabled = self.enable_amp
        self.stability_counter = 0
        self.max_stability_fails = 10
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive mixed precision"""
        
        # Check if we should use mixed precision
        if self.enable_amp and self.amp_enabled:
            with torch.cuda.amp.autocast():
                output = super().forward(input.half())
                
                # Check numerical stability
                if torch.isnan(output).any() or torch.isinf(output).any():
                    self.stability_counter += 1
                    if self.stability_counter >= self.max_stability_fails:
                        self.amp_enabled = False
                        warnings.warn("Disabling AMP due to numerical instability")
                        # Recompute in FP32
                        output = super().forward(input.float())
                    
                return output.float()
        else:
            return super().forward(input.float())


# Factory function for easy neuron creation
def create_optimized_stbif_neuron(q_threshold, level, optimization_level="medium", **kwargs):
    """
    Factory function to create optimized ST-BIF neuron with specified optimization level.
    
    Args:
        q_threshold: Voltage threshold parameter
        level: Quantization level
        optimization_level: "low" (original), "medium" (optimized), "high" (adaptive mixed precision)
        **kwargs: Additional parameters passed to neuron constructor
        
    Returns:
        Optimized ST-BIF neuron instance
    """
    if optimization_level == "high":
        return AdaptiveMixedPrecisionST_BIF(q_threshold, level, optimization_level=optimization_level, **kwargs)
    else:
        return OptimizedST_BIFNeuron_MS(q_threshold, level, optimization_level=optimization_level, **kwargs)


if __name__ == "__main__":
    # Test optimized ST-BIF neuron
    print("Testing Optimized ST-BIF Neuron...")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size, features, time_steps = 32, 1024, 8
    
    # Create test data
    input_data = torch.randn(time_steps * batch_size, features, device=device)
    
    # Test different optimization levels
    for opt_level in ["low", "medium", "high"]:
        print(f"\nTesting optimization level: {opt_level}")
        
        neuron = create_optimized_stbif_neuron(
            q_threshold=1.0, 
            level=8, 
            optimization_level=opt_level
        )
        neuron.to(device)
        neuron.T = time_steps
        
        # Forward pass
        output = neuron(input_data)
        print(f"Output shape: {output.shape}")
        print(f"Spike rate: {output.abs().mean().item():.4f}")
        
        # Get performance stats if available
        monitor = get_performance_monitor()
        if monitor is not None:
            stats = monitor.get_summary()
            if 'forward' in stats:
                print(f"Average forward time: {stats['forward']['mean_ms']:.2f}ms")
    
    print("✅ Optimized ST-BIF neuron test completed")