#!/usr/bin/env python3
"""
Debug script to analyze ST-BIF precision differences
===================================================

This script helps diagnose the numerical differences between different
precision levels in ST-BIF neuron implementations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

def create_simple_test():
    """Create a minimal test case to isolate the precision issue"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test parameters
    batch_size = 2
    time_steps = 4
    feature_dim = 4
    threshold = 1.0
    level = 8
    
    # Create deterministic input
    torch.manual_seed(42)
    x_base = torch.randn(time_steps * batch_size, feature_dim)
    
    print("=== Debugging ST-BIF Precision Differences ===")
    print(f"Input shape: {x_base.shape}")
    print(f"Input range: [{x_base.min().item():.4f}, {x_base.max().item():.4f}]")
    print(f"Input mean: {x_base.mean().item():.4f}")
    print()
    
    results = {}
    
    # Test each precision
    for dtype_name, dtype in [('fp64', torch.float64), ('fp32', torch.float32), ('fp16', torch.float16)]:
        if dtype == torch.float16 and not torch.cuda.is_available():
            continue
            
        print(f"--- Testing {dtype_name} ---")
        
        # Convert input to target precision
        x = x_base.to(dtype=dtype, device=device)
        
        # Create neuron
        torch.manual_seed(42)
        neuron = ST_BIFNeuron_MS(
            q_threshold=torch.tensor(threshold, dtype=dtype, device=device),
            level=torch.tensor(level),
            sym=False
        )
        neuron.T = time_steps
        neuron.to(device=device)
        neuron.reset()
        
        # Run forward pass
        output = neuron(x)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Spike rate: {torch.mean(torch.abs(output)).item():.4f}")
        print(f"  Non-zero elements: {torch.sum(output != 0).item()}/{output.numel()}")
        
        # Store for comparison
        results[dtype_name] = {
            'output': output.cpu().float(),  # Convert to fp32 for comparison
            'dtype': dtype,
            'device': device
        }
        print()
    
    # Compare results
    print("=== Precision Comparisons ===")
    
    if 'fp64' in results and 'fp32' in results:
        diff_64_32 = results['fp64']['output'] - results['fp32']['output']
        print(f"FP64 vs FP32:")
        print(f"  Max abs diff: {torch.max(torch.abs(diff_64_32)).item():.6f}")
        print(f"  Mean abs diff: {torch.mean(torch.abs(diff_64_32)).item():.6f}")
        print(f"  Different elements: {torch.sum(diff_64_32 != 0).item()}/{diff_64_32.numel()}")
        
        # Find where differences occur
        nonzero_diff = torch.nonzero(diff_64_32, as_tuple=False)
        if len(nonzero_diff) > 0:
            print(f"  First few differences:")
            for i in range(min(5, len(nonzero_diff))):
                idx = tuple(nonzero_diff[i].tolist())
                fp64_val = results['fp64']['output'][idx].item()
                fp32_val = results['fp32']['output'][idx].item()
                print(f"    {idx}: FP64={fp64_val:.6f}, FP32={fp32_val:.6f}, diff={diff_64_32[idx].item():.6f}")
        print()
    
    if 'fp64' in results and 'fp16' in results:
        diff_64_16 = results['fp64']['output'] - results['fp16']['output']
        print(f"FP64 vs FP16:")
        print(f"  Max abs diff: {torch.max(torch.abs(diff_64_16)).item():.6f}")
        print(f"  Mean abs diff: {torch.mean(torch.abs(diff_64_16)).item():.6f}")
        print(f"  Different elements: {torch.sum(diff_64_16 != 0).item()}/{diff_64_16.numel()}")
        print()
    
    return results

def debug_internal_states():
    """Debug internal neuron states to find where precision differences originate"""
    print("=== Debugging Internal States ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple test case
    batch_size = 1
    time_steps = 2
    feature_dim = 2
    threshold = 1.0
    level = 8
    
    # Create simple input
    x_values = torch.tensor([[0.5, 0.8], [1.2, -0.3]], dtype=torch.float32)  # 2 time steps, 2 features
    x_input = x_values.view(time_steps * batch_size, feature_dim)
    
    print(f"Input values:\n{x_values}")
    print()
    
    for dtype_name, dtype in [('fp64', torch.float64), ('fp32', torch.float32)]:
        print(f"--- {dtype_name} Internal States ---")
        
        x = x_input.to(dtype=dtype, device=device)
        
        torch.manual_seed(42)
        neuron = ST_BIFNeuron_MS(
            q_threshold=torch.tensor(threshold, dtype=dtype, device=device),
            level=torch.tensor(level),
            sym=False
        )
        neuron.T = time_steps
        neuron.to(device=device)
        neuron.reset()
        
        # Print neuron parameters
        print(f"  Threshold: {neuron.q_threshold.item():.6f} ({neuron.q_threshold.dtype})")
        print(f"  Level: {neuron.level.item()}")
        print(f"  Pos_max: {neuron.pos_max.item()}")
        print(f"  Neg_min: {neuron.neg_min.item()}")
        print(f"  Prefire: {neuron.prefire.item():.6f}")
        
        # Run forward pass
        output = neuron(x)
        
        print(f"  Output:\n{output.view(time_steps, batch_size, feature_dim)}")
        print()

def test_cuda_kernel_directly():
    """Test CUDA kernel directly to isolate kernel vs wrapper issues"""
    print("=== Testing CUDA Kernel Directly ===")
    
    from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA
    
    device = 'cuda'
    batch_size = 1
    time_steps = 2
    feature_dim = 2
    
    # Simple test input
    x_values = torch.tensor([[[0.5, 0.8]], [[1.2, -0.3]]], dtype=torch.float32, device=device)  # [T, N, F]
    
    # Parameters
    v_th = torch.tensor(1.0, device=device)
    T_max = torch.tensor(7, device=device)  # level - 1
    T_min = torch.tensor(0, device=device)
    prefire = torch.tensor(0.0, device=device)
    
    print(f"Input shape: {x_values.shape}")
    print(f"Input values:\n{x_values}")
    
    for dtype_name, dtype in [('fp64', torch.float64), ('fp32', torch.float32), ('fp16', torch.float16)]:
        print(f"\n--- {dtype_name} CUDA Kernel ---")
        
        # Convert inputs
        x = x_values.to(dtype)
        x_flat = x.flatten(2)  # [T, N*F]
        
        v_th_typed = v_th.to(dtype)
        T_max_typed = T_max.to(dtype)
        T_min_typed = T_min.to(dtype)
        prefire_typed = prefire.to(dtype)
        
        print(f"  Input flat shape: {x_flat.shape}")
        print(f"  Input flat values:\n{x_flat}")
        
        try:
            # Call CUDA kernel
            spike_seq, v_out, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(
                x_flat, v_th_typed, T_max_typed, T_min_typed, prefire_typed
            )
            
            print(f"  Spike output shape: {spike_seq.shape}")
            print(f"  Spike values:\n{spike_seq}")
            print(f"  Final voltage:\n{v_out}")
            print(f"  T sequence:\n{T_seq}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    # Run all debug tests
    print("Starting ST-BIF Precision Debug Analysis...\n")
    
    # Test 1: Simple comparison
    results = create_simple_test()
    
    # Test 2: Internal state debugging
    debug_internal_states()
    
    # Test 3: Direct CUDA kernel testing
    if torch.cuda.is_available():
        test_cuda_kernel_directly()
    else:
        print("CUDA not available, skipping direct kernel test")
    
    print("\nDebug analysis completed!")