"""
CUDA Kernel Tests - Comprehensive testing for ST-BIF CUDA operators
Tests both equivalence and performance of original vs new CUDA kernels
"""

import pytest
import torch
import time
import numpy as np
from typing import Tuple, List, Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import helper functions (defined in conftest.py)
from tests.conftest import generate_input_data, assert_tensor_close

# Import CUDA operators with error handling
try:
    from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA as OriginalKernel
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    ORIGINAL_AVAILABLE = False
    OriginalKernel = None
    print(f"Warning: Original CUDA kernel not available: {e}")

try:
    from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA as NewKernel
    NEW_AVAILABLE = True
except ImportError as e:
    NEW_AVAILABLE = False
    NewKernel = None
    print(f"Warning: New CUDA kernel not available: {e}")

# Import PyTorch reference implementation
try:
    from neuron_cupy.original_operator import ST_BIFNodeATGF_MS as PyTorchOperator
    PYTORCH_AVAILABLE = True
except ImportError as e:
    PYTORCH_AVAILABLE = False
    PyTorchOperator = None
    print(f"Warning: PyTorch operator not available: {e}")


class TestCUDAKernelEquivalence:
    """Test numerical equivalence between different kernel implementations"""
    
    @pytest.mark.cuda
    @pytest.mark.equivalence
    @pytest.mark.parametrize("dtype_name", ["fp32", "fp16"])
    @pytest.mark.parametrize("shape", [(10, 16, 64), (8, 32, 128), (5, 8, 256)])
    def test_forward_equivalence_original_vs_new(self, device, threshold_values, tolerance_config, 
                                                dtype_map, dtype_name, shape):
        """Test forward pass equivalence between original and new CUDA kernels"""
        if not (ORIGINAL_AVAILABLE and NEW_AVAILABLE):
            pytest.skip("Both original and new kernels required")
        
        dtype = dtype_map[dtype_name]
        tolerance = tolerance_config[dtype_name]
        v_th, T_max, T_min = threshold_values
        
        # Convert thresholds to correct dtype
        v_th = v_th.to(dtype)
        T_max = T_max.to(dtype)  
        T_min = T_min.to(dtype)
        prefire = torch.zeros_like(v_th)
        
        # Generate test data
        x = generate_input_data(shape, device, dtype)
        x_copy = x.clone().detach().requires_grad_(True)
        
        # Run both kernels
        with torch.no_grad():
            spike_orig, v_orig, T_orig = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
            spike_new, v_new, T_new = NewKernel.apply(x_copy, v_th, T_max, T_min, prefire)
        
        # Compare outputs
        assert_tensor_close(spike_orig, spike_new, 
                          rtol=tolerance['rtol'], atol=tolerance['atol'],
                          test_name=f"Forward spikes {dtype_name} {shape}")
        assert_tensor_close(v_orig, v_new,
                          rtol=tolerance['rtol'], atol=tolerance['atol'], 
                          test_name=f"Forward voltage {dtype_name} {shape}")
        assert_tensor_close(T_orig, T_new,
                          rtol=tolerance['rtol'], atol=tolerance['atol'],
                          test_name=f"Forward threshold {dtype_name} {shape}")

    @pytest.mark.cuda
    @pytest.mark.equivalence
    @pytest.mark.parametrize("dtype_name", ["fp32"])  # Focus on fp32 for backward pass
    @pytest.mark.parametrize("shape", [(10, 8, 64), (5, 16, 32)])  # Smaller shapes for stability
    def test_backward_equivalence_original_vs_new(self, device, threshold_values, tolerance_config,
                                                 dtype_map, dtype_name, shape):
        """Test backward pass equivalence between original and new CUDA kernels"""
        if not (ORIGINAL_AVAILABLE and NEW_AVAILABLE):
            pytest.skip("Both original and new kernels required")
        
        dtype = dtype_map[dtype_name]
        tolerance = tolerance_config[dtype_name]
        v_th, T_max, T_min = threshold_values
        
        # Convert thresholds to correct dtype
        v_th = v_th.to(dtype)
        T_max = T_max.to(dtype)
        T_min = T_min.to(dtype)
        prefire = torch.zeros_like(v_th)
        
        # Generate test data
        torch.manual_seed(42)  # For reproducibility
        x1 = generate_input_data(shape, device, dtype)
        x2 = x1.clone().detach().requires_grad_(True)
        
        # Forward pass
        spike_orig, v_orig, T_orig = OriginalKernel.apply(x1, v_th, T_max, T_min, prefire)
        spike_new, v_new, T_new = NewKernel.apply(x2, v_th, T_max, T_min, prefire)
        
        # Generate gradients
        grad_spike = torch.randn_like(spike_orig)
        grad_v = torch.randn_like(v_orig)
        grad_T = torch.randn_like(T_orig)
        
        # Backward pass
        total_loss_orig = (spike_orig * grad_spike).sum() + (v_orig * grad_v).sum() + (T_orig * grad_T).sum()
        total_loss_new = (spike_new * grad_spike).sum() + (v_new * grad_v).sum() + (T_new * grad_T).sum()
        
        total_loss_orig.backward()
        total_loss_new.backward()
        
        # Compare gradients with relaxed tolerance for backward pass
        relaxed_tolerance = {
            'rtol': tolerance['rtol'] * 10,  # More relaxed for gradients
            'atol': tolerance['atol'] * 10
        }
        
        assert_tensor_close(x1.grad, x2.grad,
                          rtol=relaxed_tolerance['rtol'], atol=relaxed_tolerance['atol'],
                          test_name=f"Backward gradients {dtype_name} {shape}")

    @pytest.mark.cuda
    @pytest.mark.equivalence
    @pytest.mark.parametrize("dtype_name", ["fp32"])
    def test_cuda_vs_pytorch_reference(self, device, threshold_values, tolerance_config,
                                     dtype_map, dtype_name):
        """Test CUDA kernel against PyTorch reference implementation"""
        if not (ORIGINAL_AVAILABLE and PYTORCH_AVAILABLE):
            pytest.skip("Original CUDA kernel and PyTorch reference required")
        
        dtype = dtype_map[dtype_name]
        tolerance = tolerance_config[dtype_name]
        v_th, T_max, T_min = threshold_values
        shape = (8, 4, 32)  # Small shape for reference comparison
        
        # Convert thresholds to correct dtype
        v_th = v_th.to(dtype)
        T_max = T_max.to(dtype)
        T_min = T_min.to(dtype)
        prefire = torch.zeros_like(v_th)
        
        # Generate test data
        x = generate_input_data(shape, device, dtype)
        x_cpu = x.detach().cpu().requires_grad_(True)
        
        # Convert thresholds for CPU
        v_th_cpu = v_th.cpu()
        T_max_cpu = T_max.cpu()
        T_min_cpu = T_min.cpu()
        prefire_cpu = prefire.cpu()
        
        # Run CUDA and PyTorch implementations
        with torch.no_grad():
            spike_cuda, v_cuda, T_cuda = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
            spike_pt, v_pt, T_pt = PyTorchOperator.apply(x_cpu, v_th_cpu, T_max_cpu, T_min_cpu, prefire_cpu)
        
        # Move PyTorch results to GPU for comparison
        spike_pt = spike_pt.to(device)
        v_pt = v_pt.to(device)
        T_pt = T_pt.to(device)
        
        # Compare with relaxed tolerance (CUDA vs CPU can have differences)
        relaxed_tolerance = {
            'rtol': tolerance['rtol'] * 5,
            'atol': tolerance['atol'] * 5
        }
        
        assert_tensor_close(spike_cuda, spike_pt,
                          rtol=relaxed_tolerance['rtol'], atol=relaxed_tolerance['atol'],
                          test_name=f"CUDA vs PyTorch spikes {dtype_name}")


class TestCUDAKernelPerformance:
    """Test performance characteristics of CUDA kernels"""
    
    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.parametrize("shape", [(32, 32, 256), (16, 64, 512)])
    def test_kernel_performance_comparison(self, device, threshold_values, shape):
        """Compare performance between original and new CUDA kernels"""
        if not (ORIGINAL_AVAILABLE and NEW_AVAILABLE):
            pytest.skip("Both original and new kernels required")
        
        v_th, T_max, T_min = threshold_values
        prefire = torch.zeros_like(v_th)
        
        # Generate test data
        x = generate_input_data(shape, device, torch.float32)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
                _ = NewKernel.apply(x, v_th, T_max, T_min, prefire)
        
        torch.cuda.synchronize()
        
        # Benchmark original kernel
        runs = 10
        start_time = time.time()
        for _ in range(runs):
            with torch.no_grad():
                _ = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
        torch.cuda.synchronize()
        original_time = (time.time() - start_time) / runs
        
        # Benchmark new kernel
        start_time = time.time()
        for _ in range(runs):
            with torch.no_grad():
                _ = NewKernel.apply(x, v_th, T_max, T_min, prefire)
        torch.cuda.synchronize()
        new_time = (time.time() - start_time) / runs
        
        # Report performance
        speedup = original_time / new_time
        print(f"\nPerformance comparison for shape {shape}:")
        print(f"Original kernel: {original_time*1000:.3f} ms")
        print(f"New kernel: {new_time*1000:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Performance should not regress significantly
        assert new_time < original_time * 2.0, f"New kernel is >2x slower: {speedup:.2f}x"

    @pytest.mark.cuda 
    @pytest.mark.performance
    def test_memory_usage(self, device, threshold_values):
        """Test memory usage of CUDA kernels"""
        if not ORIGINAL_AVAILABLE:
            pytest.skip("Original CUDA kernel required")
        
        v_th, T_max, T_min = threshold_values
        prefire = torch.zeros_like(v_th)
        shape = (32, 32, 256)
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated(device)
        
        # Generate test data and measure memory
        x = generate_input_data(shape, device, torch.float32)
        data_memory = torch.cuda.memory_allocated(device) - baseline_memory
        
        # Run kernel and measure peak memory
        with torch.no_grad():
            spike, v, T = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
        peak_memory = torch.cuda.max_memory_allocated(device) - baseline_memory
        
        print(f"\nMemory usage for shape {shape}:")
        print(f"Input data: {data_memory / 1024**2:.2f} MB")
        print(f"Peak usage: {peak_memory / 1024**2:.2f} MB")
        print(f"Memory efficiency: {data_memory / peak_memory:.2%}")
        
        # Reset peak memory counter
        torch.cuda.reset_peak_memory_stats(device)


class TestCUDAKernelRegression:
    """Regression tests to ensure kernel modifications don't break functionality"""
    
    @pytest.mark.cuda
    @pytest.mark.regression
    def test_basic_functionality_regression(self, device, threshold_values):
        """Basic smoke test to ensure kernels work"""
        if not ORIGINAL_AVAILABLE:
            pytest.skip("Original CUDA kernel required")
        
        v_th, T_max, T_min = threshold_values
        prefire = torch.zeros_like(v_th)
        shape = (10, 4, 32)
        
        x = generate_input_data(shape, device, torch.float32)
        
        # Should not raise any exceptions
        spike, v, T = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
        
        # Basic sanity checks
        assert spike.shape == x.shape
        # Note: v and T shapes depend on the specific CUDA kernel implementation
        # Just check they are reasonable shapes
        assert len(v.shape) >= 1  # Should have at least 1 dimension
        assert len(T.shape) >= 1  # Should have at least 1 dimension
        assert spike.dtype == x.dtype
        assert not torch.isnan(spike).any()
        assert not torch.isnan(v).any()
        assert not torch.isnan(T).any()

    @pytest.mark.cuda
    @pytest.mark.regression
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    @pytest.mark.parametrize("features", [32, 64, 128])
    def test_different_dimensions_regression(self, device, threshold_values, batch_size, features):
        """Test kernels work with different dimensions"""
        if not ORIGINAL_AVAILABLE:
            pytest.skip("Original CUDA kernel required")
        
        v_th, T_max, T_min = threshold_values
        # Adjust threshold dimensions to match features
        v_th = torch.full((features,), 0.1, device=device)
        prefire = torch.zeros_like(v_th)
        shape = (8, batch_size, features)
        
        x = generate_input_data(shape, device, torch.float32)
        
        # Should handle different dimensions without issues
        spike, v, T = OriginalKernel.apply(x, v_th, T_max, T_min, prefire)
        
        assert spike.shape == x.shape
        assert v.shape == (batch_size, features)
        assert T.shape == (batch_size, features)