#!/usr/bin/env python3
import sys
sys.path.append('../..')

import torch
import torch.cuda.nvtx as nvtx
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

def test_kernels():
    device = 'cuda'
    batch_size = 32
    time_steps = 8
    feature_size = 512
    
    # Generate test data
    torch.manual_seed(42)
    x_seq = torch.randn(time_steps * batch_size, feature_size, device=device)
    
    # Create neuron
    neuron = ST_BIFNeuron_MS(
        q_threshold=torch.tensor(1.0),
        level=8,
        sym=True,
        first_neuron=True
    ).to(device)
    neuron.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            neuron._reset_states()
            _ = neuron(x_seq)
    
    torch.cuda.synchronize()
    
    # Multiple runs for analysis
    nvtx.range_push("ST_BIF_Kernel_Runs")
    
    for i in range(5):
        nvtx.range_push(f"ST_BIF_Run_{i}")
        with torch.no_grad():
            neuron._reset_states()
            output = neuron(x_seq)
        torch.cuda.synchronize()
        nvtx.range_pop()
    
    nvtx.range_pop()
    
    print(f"Kernel analysis completed. Output shape: {output.shape}")

if __name__ == "__main__":
    test_kernels()
