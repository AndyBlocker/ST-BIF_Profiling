#!/usr/bin/env python3
"""
SNN Performance Profiler

This script provides comprehensive profiling for the ST-BIF SNN framework,
analyzing forward and backward pass bottlenecks using multiple profiling tools:

1. torch.profiler - PyTorch native profiling
2. nsys - NVIDIA Nsight Systems (requires external bash script)  
3. ncu - NVIDIA Nsight Compute (requires external bash script)

Usage:
    python profile/scripts/snn_profiler.py --method torch --steps 100
    python profile/scripts/snn_profiler.py --method all --warmup 50 --steps 200
"""

import sys
import os
import argparse
import time
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.profiler import profile, record_function, ProfilerActivity
import warnings

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import SNN framework components
from snn.conversion import myquan_replace_resnet
from wrapper import SNNWrapper_MS
from models import resnet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class SNNProfiler:
    """Comprehensive profiler for SNN framework"""
    
    def __init__(self, device='cuda', batch_size=32, time_steps=8, level=8, inference_mode=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.level = level
        self.inference_mode = inference_mode
        
        print(f"🔧 Initializing SNN Profiler")
        print(f"   Device: {self.device}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Time Steps: {time_steps}")
        print(f"   Quantization Level: {level}")
        print(f"   Mode: {'Inference' if inference_mode else 'Training'}")
        
        # Create output directories
        self.output_dir = "profile/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_test_data(self):
        """Create synthetic test data for profiling"""
        # CIFAR-10 style data
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        # Generate synthetic data
        data = torch.randn(self.batch_size, 3, 32, 32)
        
        # Normalize
        for i in range(3):
            data[:, i] = (data[:, i] - mean[i]) / std[i]
            
        target = torch.randint(0, 10, (self.batch_size,))
        
        return data.to(self.device), target.to(self.device)
    
    def build_snn_model(self, ann_path="checkpoints/resnet/best_ANN.pth", 
                       qann_path="checkpoints/resnet/best_QANN.pth"):
        """Build SNN model following ANN->QANN->SNN pipeline"""
        
        print("\n🏗️ Building SNN Model Pipeline")
        
        # Step 1: Create ANN model
        print("   Step 1/3: Creating ANN model...")
        ann_model = resnet.resnet18(pretrained=False)
        ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
        
        # Try loading pre-trained weights
        if os.path.exists(ann_path):
            print(f"   Loading ANN weights: {ann_path}")
            checkpoint = torch.load(ann_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                ann_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                ann_model.load_state_dict(checkpoint)
        else:
            print("   ⚠️  ANN weights not found, using random initialization")
        
        # Step 2: Convert to QANN
        print("   Step 2/3: Converting to QANN...")
        qann_model = resnet.resnet18(pretrained=False)
        qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)
        
        # Apply quantization structure
        myquan_replace_resnet(qann_model, level=self.level, weight_bit=32, is_softmax=False)
        
        # Try loading pre-trained QANN weights
        if os.path.exists(qann_path):
            print(f"   Loading QANN weights: {qann_path}")
            checkpoint = torch.load(qann_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                qann_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                qann_model.load_state_dict(checkpoint)
        else:
            print("   ⚠️  QANN weights not found, converting from ANN")
            qann_model = copy.deepcopy(ann_model)
            myquan_replace_resnet(qann_model, level=self.level, weight_bit=32, is_softmax=False)
        
        # Step 3: Convert to SNN
        print("   Step 3/3: Converting to SNN...")
        snn_model = SNNWrapper_MS(
            ann_model=qann_model,
            cfg=None,
            time_step=self.time_steps,
            Encoding_type="analog",
            level=self.level,
            neuron_type="ST-BIF",
            model_name="resnet",
            is_softmax=False,
            suppress_over_fire=False,
            record_inout=False,  # Disable I/O recording for profiling
            learnable=True,
            record_dir="/tmp/snn_profile"
        )
        
        snn_model.to(self.device)
        print("   ✅ SNN model created successfully")
        
        return snn_model
    
    def warmup_model(self, model, data, target, warmup_steps=20):
        """Warmup model to stabilize CUDA kernels"""
        print(f"\n🔥 Warming up model ({warmup_steps} steps)...")
        
        if self.inference_mode:
            model.eval()
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for i in range(warmup_steps):
                    # Forward pass only for inference
                    output = model(data)
                    loss = criterion(output, target)
                    
                    if (i + 1) % 10 == 0:
                        print(f"   Warmup step {i+1}/{warmup_steps}")
        else:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            for i in range(warmup_steps):
                optimizer.zero_grad()
                
                with torch.no_grad():
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)
                
                if (i + 1) % 10 == 0:
                    print(f"   Warmup step {i+1}/{warmup_steps}")
        
        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print("   ✅ Warmup completed")
    
    def profile_torch(self, model, data, target, steps=100):
        """Profile using torch.profiler"""
        
        print(f"\n📊 Starting PyTorch Profiler ({steps} steps)...")
        
        if self.inference_mode:
            model.eval()
            criterion = nn.CrossEntropyLoss()
        else:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
        
        # Configure profiler
        activities = [ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            
            for step in range(steps):
                if self.inference_mode:
                    with torch.no_grad():
                        with record_function("forward_pass"):
                            output = model(data)
                            loss = criterion(output, target)
                else:
                    with record_function("forward_pass"):
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                    
                    with record_function("backward_pass"):
                        loss.backward()
                        optimizer.step()
                
                if (step + 1) % 20 == 0:
                    print(f"   Profiling step {step+1}/{steps}")
        
        # Save profiler results
        timestamp = int(time.time())
        
        # Export trace for visualization
        trace_path = f"{self.output_dir}/torch_profile_{timestamp}.json"
        prof.export_chrome_trace(trace_path)
        print(f"   📁 Chrome trace saved: {trace_path}")
        
        # Export detailed results
        results_path = f"{self.output_dir}/torch_profile_{timestamp}.txt"
        with open(results_path, "w") as f:
            # Key averages table
            f.write("=== PyTorch Profiler Results ===\n\n")
            f.write("Key Averages (CPU):\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            f.write("\n\nKey Averages (CUDA):\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            
            # Memory summary if available
            if self.device.type == 'cuda':
                f.write("\n\nMemory Summary:\n")
                f.write(str(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)))
            
            # Group by input shapes
            f.write("\n\nGrouped by Input Shapes:\n")
            f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
            
            # Stack traces for top operations
            f.write("\n\nStack Traces (Top 10 CPU operations):\n")
            f.write(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
        
        print(f"   📁 Detailed results saved: {results_path}")
        
        # Print summary to console
        print("\n📈 Top 10 operations by CPU time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if self.device.type == 'cuda':
            print("\n📈 Top 10 operations by CUDA time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    def profile_memory(self, model, data, target, steps=50):
        """Profile memory usage patterns"""
        
        print(f"\n💾 Starting Memory Profiler ({steps} steps)...")
        
        if self.device.type != 'cuda':
            print("   ⚠️  Memory profiling requires CUDA")
            return
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        memory_log = []
        
        for step in range(steps):
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            forward_memory = torch.cuda.max_memory_allocated()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            backward_memory = torch.cuda.max_memory_allocated()
            
            memory_log.append({
                'step': step,
                'forward_peak': forward_memory,
                'backward_peak': backward_memory,
                'allocated': torch.cuda.memory_allocated()
            })
            
            if (step + 1) % 10 == 0:
                print(f"   Memory profiling step {step+1}/{steps}")
        
        # Save memory analysis
        timestamp = int(time.time())
        memory_path = f"{self.output_dir}/memory_profile_{timestamp}.txt"
        
        with open(memory_path, "w") as f:
            f.write("=== Memory Profiling Results ===\n\n")
            f.write(f"Initial memory: {initial_memory / 1024**2:.2f} MB\n")
            f.write(f"Peak memory during profiling: {max([m['backward_peak'] for m in memory_log]) / 1024**2:.2f} MB\n\n")
            
            f.write("Per-step memory usage (MB):\n")
            f.write("Step\tForward Peak\tBackward Peak\tAllocated\n")
            for log in memory_log:
                f.write(f"{log['step']}\t{log['forward_peak']/1024**2:.2f}\t\t"
                       f"{log['backward_peak']/1024**2:.2f}\t\t{log['allocated']/1024**2:.2f}\n")
        
        print(f"   📁 Memory analysis saved: {memory_path}")
        
        # Print summary
        avg_forward = sum([m['forward_peak'] for m in memory_log]) / len(memory_log) / 1024**2
        avg_backward = sum([m['backward_peak'] for m in memory_log]) / len(memory_log) / 1024**2
        max_memory = max([m['backward_peak'] for m in memory_log]) / 1024**2
        
        print(f"\n💾 Memory Summary:")
        print(f"   Average forward peak: {avg_forward:.2f} MB")
        print(f"   Average backward peak: {avg_backward:.2f} MB")
        print(f"   Maximum memory: {max_memory:.2f} MB")
    
    def benchmark_throughput(self, model, data, target, steps=100):
        """Benchmark training throughput"""
        
        print(f"\n⚡ Benchmarking Throughput ({steps} steps)...")
        
        if self.inference_mode:
            model.eval()
            criterion = nn.CrossEntropyLoss()
        else:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
        
        # Timing measurements
        times = []
        forward_times = []
        backward_times = []
        
        for step in range(steps):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            if self.inference_mode:
                # Forward pass timing (inference only)
                forward_start = time.time()
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, target)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # No backward pass in inference mode
                backward_start = forward_end
                backward_end = forward_end
            else:
                # Forward pass timing
                forward_start = time.time()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # Backward pass timing
                backward_start = time.time()
                loss.backward()
                optimizer.step()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_end = time.time()
            
            # Record timings
            total_time = backward_end - start_time
            forward_time = forward_end - forward_start
            backward_time = backward_end - backward_start
            
            times.append(total_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            
            if (step + 1) % 20 == 0:
                print(f"   Benchmark step {step+1}/{steps}")
        
        # Calculate statistics
        avg_total = sum(times) / len(times)
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        
        throughput = self.batch_size / avg_total
        
        # Save benchmark results
        timestamp = int(time.time())
        benchmark_path = f"{self.output_dir}/benchmark_{timestamp}.txt"
        
        with open(benchmark_path, "w") as f:
            f.write("=== SNN Throughput Benchmark ===\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Time Steps: {self.time_steps}\n")
            f.write(f"  Device: {self.device}\n\n")
            
            f.write(f"Timing Results (ms):\n")
            f.write(f"  Average total time: {avg_total*1000:.2f}\n")
            f.write(f"  Average forward time: {avg_forward*1000:.2f}\n")
            f.write(f"  Average backward time: {avg_backward*1000:.2f}\n\n")
            
            f.write(f"Throughput:\n")
            f.write(f"  Samples/sec: {throughput:.2f}\n")
            f.write(f"  Steps/sec: {1/avg_total:.2f}\n")
        
        print(f"   📁 Benchmark results saved: {benchmark_path}")
        
        # Print summary
        print(f"\n⚡ Throughput Summary:")
        print(f"   Total time: {avg_total*1000:.2f} ms/step")
        print(f"   Forward time: {avg_forward*1000:.2f} ms")
        print(f"   Backward time: {avg_backward*1000:.2f} ms")
        print(f"   Throughput: {throughput:.2f} samples/sec")


def main():
    parser = argparse.ArgumentParser(
        description='SNN Performance Profiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile/scripts/snn_profiler.py --method torch
  python profile/scripts/snn_profiler.py --method memory --steps 50
  python profile/scripts/snn_profiler.py --method benchmark --warmup 20 --steps 100
  python profile/scripts/snn_profiler.py --method all --batch-size 16
        """
    )
    
    parser.add_argument('--method', choices=['torch', 'memory', 'benchmark', 'all'], 
                       default='torch', help='Profiling method to use')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for profiling')
    parser.add_argument('--time-steps', type=int, default=8,
                       help='Time steps for SNN')
    parser.add_argument('--level', type=int, default=8,
                       help='Quantization level')
    parser.add_argument('--warmup', type=int, default=20,
                       help='Warmup steps before profiling')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of profiling steps')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--ann-path', default='checkpoints/resnet/best_ANN.pth',
                       help='Path to ANN model')
    parser.add_argument('--qann-path', default='checkpoints/resnet/best_QANN.pth',
                       help='Path to QANN model')
    parser.add_argument('--inference-mode', action='store_true',
                       help='Run in inference mode (no backward pass)')
    
    args = parser.parse_args()
    
    print("🚀 ST-BIF SNN Performance Profiler")
    print("=" * 50)
    
    # Initialize profiler
    profiler = SNNProfiler(
        device=args.device,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        level=args.level,
        inference_mode=args.inference_mode
    )
    
    # Create test data
    data, target = profiler.create_test_data()
    
    # Build SNN model
    snn_model = profiler.build_snn_model(args.ann_path, args.qann_path)
    
    # Warmup
    profiler.warmup_model(snn_model, data, target, args.warmup)
    
    # Run profiling
    if args.method == 'torch' or args.method == 'all':
        profiler.profile_torch(snn_model, data, target, args.steps)
    
    if args.method == 'memory' or args.method == 'all':
        profiler.profile_memory(snn_model, data, target, min(args.steps, 50))
    
    if args.method == 'benchmark' or args.method == 'all':
        profiler.benchmark_throughput(snn_model, data, target, args.steps)
    
    print(f"\n✅ Profiling completed! Results saved in: profile/outputs/")
    print(f"📊 View Chrome trace files at: chrome://tracing/")


if __name__ == "__main__":
    main()