#!/usr/bin/env python3
"""
Fix and verify ST-BIF precision issues
=====================================

This script identifies and fixes precision-related issues in ST-BIF neurons,
particularly focusing on the FP32 vs FP64 discrepancies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

def investigate_wrapper_vs_kernel():
    """Compare ST_BIFNeuron_MS wrapper vs direct CUDA kernel calls"""
    print("=== Wrapper vs Kernel Investigation ===")
    
    from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA
    
    device = 'cuda'
    dtype = torch.float32
    
    # Test parameters
    batch_size = 1
    time_steps = 2
    feature_dim = 2
    threshold = 1.0
    level = 8
    
    # Input
    x_values = torch.tensor([[[0.5, 0.8]], [[1.2, -0.3]]], dtype=dtype, device=device)
    
    print(f"Input: {x_values}")
    print()
    
    # Test 1: Direct CUDA kernel
    print("--- Direct CUDA Kernel ---")
    x_flat = x_values.flatten(2)
    v_th = torch.tensor(threshold, dtype=dtype, device=device)
    T_max = torch.tensor(level - 1, dtype=dtype, device=device)
    T_min = torch.tensor(0, dtype=dtype, device=device)
    prefire = torch.tensor(0.0, dtype=dtype, device=device)
    
    spike_seq, v_out, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(
        x_flat, v_th, T_max, T_min, prefire
    )
    
    print(f"Direct kernel spikes:\n{spike_seq}")
    print(f"Direct kernel final voltage: {v_out}")
    print()
    
    # Test 2: ST_BIFNeuron_MS wrapper
    print("--- ST_BIFNeuron_MS Wrapper ---")
    x_wrapper = x_values.view(time_steps * batch_size, feature_dim)
    
    torch.manual_seed(42)
    neuron = ST_BIFNeuron_MS(
        q_threshold=torch.tensor(threshold, dtype=dtype, device=device),
        level=torch.tensor(level),
        sym=False
    )
    neuron.T = time_steps
    neuron.to(device=device)
    neuron.reset()
    
    print(f"Neuron threshold: {neuron.q_threshold}")
    print(f"Neuron pos_max: {neuron.pos_max}")
    print(f"Neuron neg_min: {neuron.neg_min}")
    print(f"Neuron prefire: {neuron.prefire}")
    
    output = neuron(x_wrapper)
    
    print(f"Wrapper output:\n{output.view(time_steps, batch_size, feature_dim)}")
    print()
    
    # Compare
    wrapper_reshaped = output.view(time_steps, batch_size, feature_dim)
    kernel_reshaped = spike_seq.view(time_steps, batch_size, feature_dim) * threshold
    
    diff = wrapper_reshaped - kernel_reshaped
    print(f"Difference (wrapper - kernel):\n{diff}")
    print(f"Max abs diff: {torch.max(torch.abs(diff)).item():.6f}")

def test_tensor_precision_consistency():
    """Test if tensor precision is being maintained throughout computation"""
    print("=== Tensor Precision Consistency ===")
    
    device = 'cuda'
    
    for dtype_name, dtype in [('fp32', torch.float32), ('fp64', torch.float64)]:
        print(f"--- {dtype_name} ---")
        
        # Create test tensor and check precision maintenance
        x = torch.tensor([[1.5, 2.3]], dtype=dtype, device=device)
        threshold = torch.tensor(1.0, dtype=dtype, device=device)
        level = torch.tensor(8)
        
        print(f"Input dtype: {x.dtype}")
        print(f"Threshold dtype: {threshold.dtype}")
        
        # Create neuron
        neuron = ST_BIFNeuron_MS(
            q_threshold=threshold,
            level=level,
            sym=False
        )
        neuron.T = 1
        neuron.to(device=device)
        
        # Check internal parameter dtypes
        print(f"Neuron threshold dtype: {neuron.q_threshold.dtype}")
        print(f"Neuron pos_max dtype: {neuron.pos_max.dtype}")
        print(f"Neuron neg_min dtype: {neuron.neg_min.dtype}")
        print(f"Neuron prefire dtype: {neuron.prefire.dtype}")
        
        neuron.reset()
        output = neuron(x)
        
        print(f"Output dtype: {output.dtype}")
        print(f"Output value: {output}")
        print()

def create_precision_aware_neuron():
    """Create an improved ST-BIF neuron with better precision handling"""
    print("=== Creating Precision-Aware Neuron ===")
    
    class ST_BIFNeuron_MS_Fixed(ST_BIFNeuron_MS):
        """ST-BIF Neuron with improved precision handling"""
        
        def __init__(self, q_threshold, level, sym=False, first_neuron=False, need_spike_tracer=False):
            super().__init__(q_threshold, level, sym, first_neuron, need_spike_tracer)
        
        def to(self, device=None, dtype=None, non_blocking=False):
            """Override to ensure all parameters maintain consistent precision"""
            result = super().to(device=device, dtype=dtype, non_blocking=non_blocking)
            
            if dtype is not None:
                # Ensure all buffer tensors are in the correct dtype
                if hasattr(self, 'pos_max'):
                    self.pos_max = self.pos_max.to(dtype=dtype)
                if hasattr(self, 'neg_min'):
                    self.neg_min = self.neg_min.to(dtype=dtype)
                if hasattr(self, 'prefire'):
                    self.prefire = self.prefire.to(dtype=dtype)
                # Keep q_threshold as parameter, it should auto-convert
                
            return result
        
        def forward(self, input):
            """Forward pass with explicit dtype checking"""
            # Ensure all parameters match input dtype
            input_dtype = input.dtype
            
            if self.q_threshold.dtype != input_dtype:
                print(f"Warning: q_threshold dtype {self.q_threshold.dtype} != input dtype {input_dtype}")
            
            if self.pos_max.dtype != input_dtype:
                self.pos_max = self.pos_max.to(dtype=input_dtype)
            
            if self.neg_min.dtype != input_dtype:
                self.neg_min = self.neg_min.to(dtype=input_dtype)
                
            if self.prefire.dtype != input_dtype:
                self.prefire = self.prefire.to(dtype=input_dtype)
            
            return super().forward(input)
    
    # Test the fixed neuron
    device = 'cuda'
    
    for dtype_name, dtype in [('fp32', torch.float32), ('fp64', torch.float64)]:
        print(f"--- Testing Fixed Neuron with {dtype_name} ---")
        
        x = torch.tensor([[1.5, 2.3]], dtype=dtype, device=device)
        
        neuron = ST_BIFNeuron_MS_Fixed(
            q_threshold=torch.tensor(1.0, dtype=dtype, device=device),
            level=torch.tensor(8),
            sym=False
        )
        neuron.T = 1
        neuron.to(device=device, dtype=dtype)
        neuron.reset()
        
        output = neuron(x)
        
        print(f"Input dtype: {x.dtype}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output value: {output}")
        print(f"Threshold dtype: {neuron.q_threshold.dtype}")
        print(f"Pos_max dtype: {neuron.pos_max.dtype}")
        print()

def comprehensive_precision_test():
    """Run comprehensive test across all precision scenarios"""
    print("=== Comprehensive Precision Test ===")
    
    device = 'cuda'
    
    # Test parameters
    test_cases = [
        {'input_val': 0.8, 'threshold': 1.0, 'expected_spikes': 0},
        {'input_val': 1.5, 'threshold': 1.0, 'expected_spikes': 1},
        {'input_val': -0.5, 'threshold': 1.0, 'expected_spikes': 0},
    ]
    
    results = {}
    
    for dtype_name, dtype in [('fp16', torch.float16), ('fp32', torch.float32), ('fp64', torch.float64)]:
        print(f"--- {dtype_name} Results ---")
        results[dtype_name] = []
        
        for i, case in enumerate(test_cases):
            x = torch.tensor([[case['input_val']]], dtype=dtype, device=device)
            
            neuron = ST_BIFNeuron_MS(
                q_threshold=torch.tensor(case['threshold'], dtype=dtype, device=device),
                level=torch.tensor(8),
                sym=False
            )
            neuron.T = 1
            neuron.to(device=device)
            neuron.reset()
            
            output = neuron(x)
            spike_count = torch.sum(torch.abs(output)).item()
            
            print(f"  Case {i+1}: input={case['input_val']:.1f}, spikes={spike_count:.1f}")
            results[dtype_name].append(spike_count)
        
        print()
    
    # Compare results
    print("--- Cross-Precision Comparison ---")
    for i, case in enumerate(test_cases):
        print(f"Case {i+1} (input={case['input_val']:.1f}):")
        for dtype_name in results:
            print(f"  {dtype_name}: {results[dtype_name][i]:.1f}")
        
        # Check consistency
        values = list(results[dtype_name][i] for dtype_name in results)
        if len(set(values)) == 1:
            print("  ✓ All precisions consistent")
        else:
            print("  ✗ Precision inconsistency detected!")
        print()

if __name__ == "__main__":
    print("Starting ST-BIF Precision Fix and Verification...\n")
    
    if torch.cuda.is_available():
        # Run all tests
        investigate_wrapper_vs_kernel()
        test_tensor_precision_consistency()
        create_precision_aware_neuron()
        comprehensive_precision_test()
    else:
        print("CUDA not available, skipping tests")
    
    print("Precision analysis completed!")