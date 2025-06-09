#!/usr/bin/env python3
"""
Deep analysis of ST-BIF precision differences
=============================================

This script provides a detailed analysis of the numerical differences
between different precision levels in ST-BIF CUDA kernels.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS
from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA

def analyze_cuda_kernel_precision():
    """Analyze CUDA kernel behavior with identical inputs across precisions"""
    print("=== Deep CUDA Kernel Precision Analysis ===")
    
    device = 'cuda'
    
    # Use simple, controlled inputs
    simple_inputs = [
        torch.tensor([[[0.5]]], device=device),      # Below threshold
        torch.tensor([[[1.5]]], device=device),      # Above threshold  
        torch.tensor([[[0.0]]], device=device),      # Zero input
        torch.tensor([[[-0.5]]], device=device),     # Negative input
    ]
    
    for i, x_base in enumerate(simple_inputs):
        print(f"\n--- Test Case {i+1}: Input = {x_base.item():.1f} ---")
        
        # Parameters
        v_th = torch.tensor(1.0, device=device)
        T_max = torch.tensor(7, device=device)  # level - 1
        T_min = torch.tensor(0, device=device)
        prefire = torch.tensor(0.0, device=device)
        
        results = {}
        
        for dtype_name, dtype in [('FP64', torch.float64), ('FP32', torch.float32), ('FP16', torch.float16)]:
            print(f"\n{dtype_name}:")
            
            # Convert all inputs to target precision
            x = x_base.to(dtype)
            x_flat = x.flatten(2)
            
            v_th_typed = v_th.to(dtype)
            T_max_typed = T_max.to(dtype)
            T_min_typed = T_min.to(dtype)
            prefire_typed = prefire.to(dtype)
            
            print(f"  Input: {x_flat.item():.6f} ({x_flat.dtype})")
            print(f"  v_th: {v_th_typed.item():.6f}")
            print(f"  T_max: {T_max_typed.item():.6f}")
            print(f"  T_min: {T_min_typed.item():.6f}")
            print(f"  prefire: {prefire_typed.item():.6f}")
            
            try:
                spike_seq, v_out, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(
                    x_flat, v_th_typed, T_max_typed, T_min_typed, prefire_typed
                )
                
                print(f"  → Spike: {spike_seq.item():.6f}")
                print(f"  → Final V: {v_out.item():.6f}")
                print(f"  → T_seq: {T_seq.item():.6f}")
                
                results[dtype_name] = {
                    'spike': spike_seq.item(),
                    'v_out': v_out.item(),
                    'T_seq': T_seq.item()
                }
                
            except Exception as e:
                print(f"  → Error: {e}")
                results[dtype_name] = {'error': str(e)}
        
        # Compare results
        print(f"\nComparison for input {x_base.item():.1f}:")
        
        if all('error' not in r for r in results.values()):
            # Check spike consistency
            spikes = [results[p]['spike'] for p in results.keys()]
            if len(set(spikes)) == 1:
                print(f"  ✓ Spikes consistent: {spikes[0]}")
            else:
                print(f"  ❌ Spike inconsistency: {dict(zip(results.keys(), spikes))}")
            
            # Check voltage consistency
            voltages = [results[p]['v_out'] for p in results.keys()]
            v_diffs = [abs(v - voltages[0]) for v in voltages[1:]]
            max_v_diff = max(v_diffs) if v_diffs else 0
            if max_v_diff < 1e-5:
                print(f"  ✓ Voltages consistent (max diff: {max_v_diff:.2e})")
            else:
                print(f"  ⚠️  Voltage differences: {dict(zip(results.keys(), voltages))}")

def analyze_intermediate_states():
    """Analyze what happens inside the CUDA kernel at each step"""
    print("\n=== Intermediate State Analysis ===")
    
    # We'll create a simplified version of the ST-BIF logic to trace execution
    device = 'cuda'
    
    # Test parameters
    x_values = torch.tensor([0.8, 1.2], device=device)  # Two time steps
    v_th = 1.0
    T_max = 7
    T_min = 0
    prefire = 0.0
    
    print(f"Tracing ST-BIF logic for inputs: {x_values.tolist()}")
    print(f"Parameters: v_th={v_th}, T_max={T_max}, T_min={T_min}, prefire={prefire}")
    
    for dtype_name, dtype in [('FP64', torch.float64), ('FP32', torch.float32)]:
        print(f"\n--- {dtype_name} Trace ---")
        
        # Convert to target precision
        x = x_values.to(dtype)
        v_th_f = torch.tensor(v_th, dtype=dtype, device=device)
        T_max_f = torch.tensor(T_max, dtype=dtype, device=device)
        T_min_f = torch.tensor(T_min, dtype=dtype, device=device)
        prefire_f = torch.tensor(prefire, dtype=dtype, device=device)
        
        # Manual ST-BIF simulation (following the CUDA kernel logic)
        v = torch.tensor(0.5, dtype=dtype, device=device) * v_th_f + prefire_f * v_th_f
        T = torch.tensor(0.0, dtype=dtype, device=device)
        
        print(f"Initial state: v={v.item():.6f}, T={T.item():.6f}")
        
        for t, x_t in enumerate(x):
            print(f"\nTime step {t}: input={x_t.item():.6f}")
            
            # Update voltage
            H = v + x_t
            print(f"  H (v + input) = {H.item():.6f}")
            
            # Check spike conditions
            pos_condition = (H >= v_th_f) and (T < T_max_f)
            neg_condition = (H < 0) and (T > T_min_f)
            
            print(f"  Positive spike condition: H >= v_th AND T < T_max = {H.item():.6f} >= {v_th_f.item():.6f} AND {T.item():.6f} < {T_max_f.item():.6f} = {pos_condition}")
            print(f"  Negative spike condition: H < 0 AND T > T_min = {H.item():.6f} < 0 AND {T.item():.6f} > {T_min_f.item():.6f} = {neg_condition}")
            
            # Determine spike
            if pos_condition:
                spike = 1.0
            elif neg_condition:
                spike = -1.0
            else:
                spike = 0.0
            
            print(f"  → Spike: {spike}")
            
            # Update state
            v = H - v_th_f * spike
            T = T + spike
            
            print(f"  → New v: {v.item():.6f}")
            print(f"  → New T: {T.item():.6f}")

def test_gradient_computation():
    """Test gradient computation across precisions"""
    print("\n=== Gradient Computation Analysis ===")
    
    device = 'cuda'
    
    # Simple test case
    batch_size = 1
    time_steps = 2
    feature_dim = 1
    
    x_base = torch.tensor([[0.8], [1.2]], device=device)  # [T*B, F]
    
    for dtype_name, dtype in [('FP64', torch.float64), ('FP32', torch.float32)]:
        print(f"\n--- {dtype_name} Gradient Test ---")
        
        x = x_base.to(dtype=dtype)
        x.requires_grad_(True)
        
        neuron = ST_BIFNeuron_MS(
            q_threshold=torch.tensor(1.0, dtype=dtype, device=device),
            level=torch.tensor(8),
            sym=False
        )
        neuron.T = time_steps
        neuron.to(device=device)
        neuron.reset()
        
        # Forward pass
        output = neuron(x)
        print(f"  Output: {output.flatten().tolist()}")
        
        # Compute loss and gradients
        loss = torch.sum(output ** 2)
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        if x.grad is not None:
            print(f"  Gradients: {x.grad.flatten().tolist()}")
            print(f"  Grad norm: {torch.norm(x.grad).item():.6f}")
        else:
            print(f"  Gradients: None")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Deep ST-BIF Precision Analysis")
        print("=" * 50)
        
        analyze_cuda_kernel_precision()
        analyze_intermediate_states()
        test_gradient_computation()
        
        print("\n" + "=" * 50)
        print("Analysis completed!")
    else:
        print("CUDA not available, skipping analysis")