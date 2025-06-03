#!/usr/bin/env python3
"""
Fixed SNN Optimization Utilities - 修复版本

This module provides FIXED optimization utilities for ST-BIF SNN that avoid performance degradation.
修复了导致性能退化的关键问题。

Key fixes:
1. 选择性优化而非全量优化 - Selective optimizations instead of aggressive ones
2. ST-BIF兼容的配置 - ST-BIF compatible configurations  
3. 正确的时机控制 - Proper timing control
4. 保守的内存管理 - Conservative memory management
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, Union
import threading
import warnings
from contextlib import contextmanager

class CompatibleTensorMemoryPool:
    """
    ST-BIF兼容的内存池 - ST-BIF compatible memory pool
    更保守的设置以避免与现有tensor管理冲突
    """
    
    def __init__(self, device='cuda', max_pool_size=5):
        self.device = torch.device(device)
        self.pool = {}
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
        self.max_pool_size = max_pool_size  # 减少池大小以避免内存问题
        self.enabled = True
        
    def enable(self):
        """启用内存池"""
        self.enabled = True
        
    def disable(self):
        """禁用内存池（回退到标准分配）"""
        self.enabled = False
        self.clear_pool()
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   requires_grad: bool = False) -> torch.Tensor:
        """
        获取tensor，如果禁用则直接创建新的
        """
        if not self.enabled:
            return torch.zeros(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
            
        key = (shape, dtype, requires_grad)
        
        with self.lock:
            if key in self.pool and len(self.pool[key]) > 0:
                tensor = self.pool[key].pop()
                self.hit_count += 1
                tensor.zero_()
                return tensor
            else:
                self.miss_count += 1
                return torch.zeros(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        返回tensor到池中，如果禁用则忽略
        """
        if not self.enabled or tensor.device != self.device:
            return
            
        key = (tuple(tensor.shape), tensor.dtype, tensor.requires_grad)
        
        with self.lock:
            if key not in self.pool:
                self.pool[key] = []
            
            # 使用更小的池大小
            if len(self.pool[key]) < self.max_pool_size:
                tensor = tensor.detach()
                if tensor.grad is not None:
                    tensor.grad = None
                self.pool[key].append(tensor)
    
    def clear_pool(self):
        """清空内存池"""
        with self.lock:
            self.pool.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'enabled': self.enabled,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'pool_sizes': {str(k): len(v) for k, v in self.pool.items()}
        }

class ConservativeCUDAOptimizer:
    """
    保守的CUDA优化器 - Conservative CUDA optimizer
    只应用与ST-BIF兼容的优化
    """
    
    @staticmethod
    def setup_st_bif_cuda_optimizations(enable_tf32=True, enable_cudnn_benchmark=None, 
                                      memory_fraction=0.85):
        """
        为ST-BIF设置保守的CUDA优化
        
        Args:
            enable_tf32: 启用TF32（对ST-BIF安全）
            enable_cudnn_benchmark: 启用cuDNN基准测试（None=自动检测）
            memory_fraction: GPU内存使用比例
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, skipping optimizations")
            return
        
        optimizations_applied = []
        
        # 1. TF32优化（对ST-BIF安全）
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations_applied.append("TF32 enabled")
        
        # 2. 智能cuDNN基准测试设置
        if enable_cudnn_benchmark is None:
            # 自动检测：训练时启用，推理时禁用
            enable_cudnn_benchmark = torch.is_grad_enabled()
            
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark
        if enable_cudnn_benchmark:
            optimizations_applied.append("cuDNN benchmark enabled")
        else:
            optimizations_applied.append("cuDNN benchmark disabled (for variable inputs)")
        
        # 3. 保守的内存设置
        try:
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(memory_fraction)
                optimizations_applied.append(f"Memory fraction: {memory_fraction}")
        except Exception as e:
            warnings.warn(f"Could not set memory fraction: {e}")
        
        # 4. 清理现有缓存
        torch.cuda.empty_cache()
        optimizations_applied.append("Memory cache cleared")
        
        print("🔧 ST-BIF CUDA Optimizations Applied:")
        for opt in optimizations_applied:
            print(f"   ✅ {opt}")
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved, 
            'max_reserved_gb': max_reserved
        }
    
    @staticmethod
    def clear_memory_cache():
        """清理CUDA内存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class ST_BIF_OptimizationManager:
    """
    ST-BIF专用优化管理器 - ST-BIF specific optimization manager
    解决性能退化问题的核心类
    """
    
    def __init__(self):
        self.memory_pool = None
        self.is_optimized = False
        self.optimization_config = {}
        
    def setup_optimizations(self, 
                          enable_memory_pool=True,
                          enable_tf32=True, 
                          enable_cudnn_benchmark=None,
                          memory_pool_size=3,
                          verbose=True):
        """
        设置ST-BIF兼容的优化
        
        Args:
            enable_memory_pool: 启用内存池
            enable_tf32: 启用TF32
            enable_cudnn_benchmark: 启用cuDNN基准测试（None=自动）
            memory_pool_size: 内存池大小
            verbose: 显示详细信息
        """
        if self.is_optimized:
            if verbose:
                print("⚠️  Optimizations already applied")
            return self
        
        if verbose:
            print("🚀 Setting up ST-BIF compatible optimizations...")
        
        # 1. CUDA优化
        ConservativeCUDAOptimizer.setup_st_bif_cuda_optimizations(
            enable_tf32=enable_tf32,
            enable_cudnn_benchmark=enable_cudnn_benchmark
        )
        
        # 2. 内存池（如果启用）
        if enable_memory_pool:
            self.memory_pool = CompatibleTensorMemoryPool(max_pool_size=memory_pool_size)
            if verbose:
                print(f"   ✅ Memory pool initialized (max_size={memory_pool_size})")
        
        # 保存配置
        self.optimization_config = {
            'memory_pool': enable_memory_pool,
            'tf32': enable_tf32,
            'cudnn_benchmark': enable_cudnn_benchmark,
            'pool_size': memory_pool_size
        }
        
        self.is_optimized = True
        
        if verbose:
            print("✅ ST-BIF optimizations setup complete")
            print("⚠️  Note: Mixed precision disabled (incompatible with ST-BIF)")
            
        return self
    
    def get_memory_pool(self) -> Optional[CompatibleTensorMemoryPool]:
        """获取内存池实例"""
        return self.memory_pool
    
    def enable_memory_pool(self):
        """启用内存池"""
        if self.memory_pool:
            self.memory_pool.enable()
            print("✅ Memory pool enabled")
    
    def disable_memory_pool(self):
        """禁用内存池（用于调试性能问题）"""
        if self.memory_pool:
            self.memory_pool.disable()
            print("⚠️  Memory pool disabled")
    
    def get_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        status = {
            'is_optimized': self.is_optimized,
            'config': self.optimization_config.copy(),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            status['cuda_info'] = ConservativeCUDAOptimizer.get_memory_info()
            status['cuda_settings'] = {
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                'tf32_cudnn': torch.backends.cudnn.allow_tf32
            }
        
        if self.memory_pool:
            status['memory_pool'] = self.memory_pool.get_stats()
            
        return status
    
    def cleanup(self):
        """清理优化资源"""
        if self.memory_pool:
            self.memory_pool.clear_pool()
        ConservativeCUDAOptimizer.clear_memory_cache()
        print("🧹 Optimization cleanup complete")

# 全局实例
_global_st_bif_optimizer = None

def get_st_bif_optimizer() -> ST_BIF_OptimizationManager:
    """获取全局ST-BIF优化器实例"""
    global _global_st_bif_optimizer
    if _global_st_bif_optimizer is None:
        _global_st_bif_optimizer = ST_BIF_OptimizationManager()
    return _global_st_bif_optimizer

# 便捷函数
def setup_st_bif_optimizations(**kwargs):
    """
    便捷函数：设置ST-BIF优化
    使用示例：
    optimizer = setup_st_bif_optimizations(enable_memory_pool=True)
    """
    return get_st_bif_optimizer().setup_optimizations(**kwargs)

def cleanup_st_bif_optimizations():
    """清理ST-BIF优化"""
    get_st_bif_optimizer().cleanup()

def get_st_bif_memory_pool():
    """获取ST-BIF内存池"""
    return get_st_bif_optimizer().get_memory_pool()

if __name__ == "__main__":
    # 测试修复版本的优化工具
    print("Testing Fixed ST-BIF Optimization Utilities...")
    print("=" * 50)
    
    # 1. 设置优化
    optimizer = setup_st_bif_optimizations(
        enable_memory_pool=True,
        enable_tf32=True,
        memory_pool_size=3,
        verbose=True
    )
    
    # 2. 测试内存池
    memory_pool = get_st_bif_memory_pool()
    if memory_pool:
        print("\n📊 Testing memory pool...")
        tensor1 = memory_pool.get_tensor((100, 100))
        memory_pool.return_tensor(tensor1)
        tensor2 = memory_pool.get_tensor((100, 100))  # Should reuse
        
        print("Memory pool stats:", memory_pool.get_stats())
    
    # 3. 显示状态
    print("\n📋 Optimization status:")
    status = optimizer.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # 4. 清理
    cleanup_st_bif_optimizations()
    print("\n✅ Test completed successfully")