#!/usr/bin/env python3
"""
SNN Optimization Utilities

This module provides optimization utilities for ST-BIF SNN including:
1. Memory pool for reducing allocations
2. CUDA optimization settings  
3. Mixed precision training utilities
4. Performance monitoring tools
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import threading
import warnings
from contextlib import contextmanager

class TensorMemoryPool:
    """
    Memory pool for reducing frequent tensor allocations/deallocations.
    Reduces GPU memory fragmentation and improves performance.
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.pool = {}
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   requires_grad: bool = False) -> torch.Tensor:
        """
        Get a tensor from pool or create new one if not available.
        
        Args:
            shape: Tensor shape tuple
            dtype: Tensor data type  
            requires_grad: Whether tensor requires gradient
            
        Returns:
            Tensor from pool or newly created
        """
        key = (shape, dtype, requires_grad)
        
        with self.lock:
            if key in self.pool and len(self.pool[key]) > 0:
                tensor = self.pool[key].pop()
                self.hit_count += 1
                # Reset tensor to zero to ensure clean state
                tensor.zero_()
                return tensor
            else:
                self.miss_count += 1
                tensor = torch.zeros(shape, dtype=dtype, device=self.device, 
                                   requires_grad=requires_grad)
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if tensor.device != self.device:
            return
            
        key = (tuple(tensor.shape), tensor.dtype, tensor.requires_grad)
        
        with self.lock:
            if key not in self.pool:
                self.pool[key] = []
            
            # Limit pool size to prevent excessive memory usage
            if len(self.pool[key]) < 10:
                # Detach tensor and clear gradients
                tensor = tensor.detach()
                if tensor.grad is not None:
                    tensor.grad = None
                self.pool[key].append(tensor)
    
    def clear_pool(self):
        """Clear all tensors from memory pool."""
        with self.lock:
            self.pool.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'pool_sizes': {str(k): len(v) for k, v in self.pool.items()}
        }

# Global memory pool instance
_global_memory_pool = None

def get_memory_pool(device='cuda') -> TensorMemoryPool:
    """Get global memory pool instance."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = TensorMemoryPool(device)
    return _global_memory_pool

class CUDAOptimizer:
    """
    CUDA optimization utilities for improving GPU performance.
    """
    
    @staticmethod
    def setup_cuda_optimizations():
        """
        Setup CUDA optimizations for better performance.
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, skipping CUDA optimizations")
            return
        
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic mode if needed (can reduce performance)
        # torch.backends.cudnn.deterministic = True
        
        # Optimize memory allocator
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory fraction to prevent OOM
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.9)
        
        print("✅ CUDA optimizations enabled:")
        print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   - Memory pool initialized")
    
    @staticmethod
    def clear_memory_cache():
        """Clear CUDA memory cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved, 
            'max_reserved_gb': max_reserved
        }

class MixedPrecisionTrainer:
    """
    Mixed precision training utilities for SNN models.
    """
    
    def __init__(self, enabled: bool = True, init_scale: float = 2.**16):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale) if self.enabled else None
        
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer, model_params=None):
        """Step optimizer with gradient scaling."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with proper scaling."""
        if self.enabled and self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

class PerformanceMonitor:
    """
    Performance monitoring utilities for tracking optimization effects.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all performance counters."""
        self.forward_times = []
        self.backward_times = []
        self.memory_usage = []
        self.kernel_launches = 0
    
    def record_forward_time(self, time_ms: float):
        """Record forward pass time."""
        self.forward_times.append(time_ms)
    
    def record_backward_time(self, time_ms: float):
        """Record backward pass time."""
        self.backward_times.append(time_ms)
    
    def record_memory_usage(self, usage_gb: float):
        """Record memory usage."""
        self.memory_usage.append(usage_gb)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        import statistics
        
        summary = {}
        
        if self.forward_times:
            summary['forward'] = {
                'mean_ms': statistics.mean(self.forward_times),
                'std_ms': statistics.stdev(self.forward_times) if len(self.forward_times) > 1 else 0,
                'min_ms': min(self.forward_times),
                'max_ms': max(self.forward_times)
            }
        
        if self.backward_times:
            summary['backward'] = {
                'mean_ms': statistics.mean(self.backward_times),
                'std_ms': statistics.stdev(self.backward_times) if len(self.backward_times) > 1 else 0,
                'min_ms': min(self.backward_times),
                'max_ms': max(self.backward_times)
            }
        
        if self.memory_usage:
            summary['memory'] = {
                'mean_gb': statistics.mean(self.memory_usage),
                'peak_gb': max(self.memory_usage)
            }
        
        return summary

# Global performance monitor
_global_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _global_monitor

# Convenience functions for easy usage
def setup_optimizations():
    """Setup all available optimizations."""
    CUDAOptimizer.setup_cuda_optimizations()
    return get_memory_pool(), MixedPrecisionTrainer()

def cleanup_optimizations():
    """Cleanup and clear optimization resources."""
    get_memory_pool().clear_pool()
    CUDAOptimizer.clear_memory_cache()
    get_performance_monitor().reset()

if __name__ == "__main__":
    # Test optimization utilities
    print("Testing SNN Optimization Utilities...")
    
    # Setup optimizations
    memory_pool, mp_trainer = setup_optimizations()
    
    # Test memory pool
    tensor1 = memory_pool.get_tensor((1000, 1000))
    memory_pool.return_tensor(tensor1)
    tensor2 = memory_pool.get_tensor((1000, 1000))  # Should reuse tensor1
    
    print("Memory pool stats:", memory_pool.get_stats())
    print("CUDA memory info:", CUDAOptimizer.get_memory_info())
    
    # Cleanup
    cleanup_optimizations()
    print("✅ Optimization utilities test completed")