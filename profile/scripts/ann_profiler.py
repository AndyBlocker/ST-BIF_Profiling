#!/usr/bin/env python3
"""
ANN Performance Profiler

This script provides comprehensive profiling for standard ANN models,
serving as a baseline comparison for SNN performance analysis.

Scenarios:
1. Offline Testing (batch_size=32) - Simulates batch processing
2. Online Streaming (batch_size=1) - Simulates real-time inference

Usage:
    python profile/scripts/ann_profiler.py --scenario offline
    python profile/scripts/ann_profiler.py --scenario online
    python profile/scripts/ann_profiler.py --scenario both
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

# Import model
from models import resnet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ANNProfiler:
    """Comprehensive profiler for standard ANN models"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing ANN Profiler")
        print(f"   Device: {self.device}")
        
        # Create output directories
        self.output_dir = "profile/outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_test_data(self, batch_size):
        """Create synthetic test data for profiling"""
        # CIFAR-10 style data
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        # Generate synthetic data
        data = torch.randn(batch_size, 3, 32, 32)
        
        # Normalize
        for i in range(3):
            data[:, i] = (data[:, i] - mean[i]) / std[i]
            
        target = torch.randint(0, 10, (batch_size,))
        
        return data.to(self.device), target.to(self.device)
    
    def build_ann_model(self, ann_path="checkpoints/resnet/best_ANN.pth"):
        """Build standard ANN model (ResNet18)"""
        
        print("\n🏗️ Building ANN Model")
        
        # Create ANN model
        print("   Creating ResNet18 model...")
        ann_model = resnet.resnet18(pretrained=False)
        ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
        
        # Try loading pre-trained weights
        if os.path.exists(ann_path):
            print(f"   Loading ANN weights: {ann_path}")
            checkpoint = torch.load(ann_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                ann_model.load_state_dict(checkpoint['model_state_dict'])
                if 'best_acc' in checkpoint:
                    print(f"   Training best accuracy: {checkpoint['best_acc']:.2f}%")
            else:
                ann_model.load_state_dict(checkpoint)
        else:
            print("   ⚠️  ANN weights not found, using random initialization")
        
        ann_model.to(self.device)
        print("   ✅ ANN model created successfully")
        
        return ann_model
    
    def warmup_model(self, model, data, target, warmup_steps=20):
        """Warmup model to stabilize CUDA kernels"""
        print(f"\n🔥 Warming up model ({warmup_steps} steps)...")
        
        # Set model mode based on batch size to avoid BatchNorm issues
        batch_size = data.size(0)
        if batch_size == 1:
            model.eval()
            print("   ⚠️  Using eval() mode for batch_size=1 to avoid BatchNorm issues")
        else:
            model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for i in range(warmup_steps):
            if batch_size == 1:
                # Inference-only warmup for batch_size=1
                with torch.no_grad():
                    output = model(data)
            else:
                # Training warmup for batch_size > 1
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"   Warmup step {i+1}/{warmup_steps}")
        
        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        print("   ✅ Warmup completed")
    
    def profile_torch(self, model, data, target, steps, scenario_name, batch_size):
        """Profile using torch.profiler"""
        
        print(f"\n📊 Starting PyTorch Profiler for {scenario_name} ({steps} steps)...")
        
        # Set model mode based on batch size to avoid BatchNorm issues
        if batch_size == 1:
            model.eval()
            print("   ⚠️  Using eval() mode for batch_size=1 to avoid BatchNorm issues")
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
                if batch_size == 1:
                    # Inference-only profiling for batch_size=1
                    with record_function("inference_pass"):
                        with torch.no_grad():
                            output = model(data)
                else:
                    # Training profiling for batch_size > 1
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
        trace_path = f"{self.output_dir}/ann_torch_profile_{scenario_name}_bs{batch_size}_{timestamp}.json"
        prof.export_chrome_trace(trace_path)
        print(f"   📁 Chrome trace saved: {trace_path}")
        
        # Export detailed results
        results_path = f"{self.output_dir}/ann_torch_profile_{scenario_name}_bs{batch_size}_{timestamp}.txt"
        with open(results_path, "w") as f:
            # Header
            f.write(f"=== ANN PyTorch Profiler Results ===\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Key averages table
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
        
        print(f"   📁 Detailed results saved: {results_path}")
        
        # Print summary to console
        print(f"\n📈 Top 10 operations by CPU time ({scenario_name}):")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if self.device.type == 'cuda':
            print(f"\n📈 Top 10 operations by CUDA time ({scenario_name}):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    def profile_memory(self, model, data, target, steps, scenario_name, batch_size):
        """Profile memory usage patterns"""
        
        print(f"\n💾 Starting Memory Profiler for {scenario_name} ({steps} steps)...")
        
        if self.device.type != 'cuda':
            print("   ⚠️  Memory profiling requires CUDA")
            return
        
        # Set model mode based on batch size to avoid BatchNorm issues
        if batch_size == 1:
            model.eval()
            print("   ⚠️  Using eval() mode for batch_size=1 to avoid BatchNorm issues")
        else:
            model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated()
        memory_log = []
        
        for step in range(steps):
            torch.cuda.reset_peak_memory_stats()
            
            if batch_size == 1:
                # Inference-only memory profiling for batch_size=1
                with torch.no_grad():
                    output = model(data)
                
                forward_memory = torch.cuda.max_memory_allocated()
                backward_memory = forward_memory  # No backward pass
                
            else:
                # Training memory profiling for batch_size > 1
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
        memory_path = f"{self.output_dir}/ann_memory_profile_{scenario_name}_bs{batch_size}_{timestamp}.txt"
        
        with open(memory_path, "w") as f:
            f.write(f"=== ANN Memory Profiling Results ===\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Batch Size: {batch_size}\n\n")
            f.write(f"Initial memory: {initial_memory / 1024**2:.2f} MB\n")
            f.write(f"Peak memory during profiling: {max([m['backward_peak'] for m in memory_log]) / 1024**2:.2f} MB\n\n")
            
            f.write("Per-step memory usage (MB):\n")
            f.write("Step\tForward Peak\tBackward Peak\tAllocated\n")
            for log in memory_log[:10]:  # Show first 10 steps
                f.write(f"{log['step']}\t{log['forward_peak']/1024**2:.2f}\t\t"
                       f"{log['backward_peak']/1024**2:.2f}\t\t{log['allocated']/1024**2:.2f}\n")
        
        print(f"   📁 Memory analysis saved: {memory_path}")
        
        # Print summary
        avg_forward = sum([m['forward_peak'] for m in memory_log]) / len(memory_log) / 1024**2
        avg_backward = sum([m['backward_peak'] for m in memory_log]) / len(memory_log) / 1024**2
        max_memory = max([m['backward_peak'] for m in memory_log]) / 1024**2
        
        print(f"\n💾 Memory Summary ({scenario_name}):")
        print(f"   Average forward peak: {avg_forward:.2f} MB")
        print(f"   Average backward peak: {avg_backward:.2f} MB")
        print(f"   Maximum memory: {max_memory:.2f} MB")
    
    def benchmark_throughput(self, model, data, target, steps, scenario_name, batch_size):
        """Benchmark training throughput"""
        
        print(f"\n⚡ Benchmarking Throughput for {scenario_name} ({steps} steps)...")
        
        # Set model mode based on batch size to avoid BatchNorm issues
        if batch_size == 1:
            model.eval()
            print("   ⚠️  Using eval() mode for batch_size=1 to avoid BatchNorm issues")
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
            
            if batch_size == 1:
                # Inference-only timing for batch_size=1
                forward_start = time.time()
                with torch.no_grad():
                    output = model(data)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                forward_end = time.time()
                
                # No backward pass
                backward_start = forward_end
                backward_end = forward_end
                
            else:
                # Training timing for batch_size > 1
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
        
        throughput = batch_size / avg_total
        
        # Save benchmark results
        timestamp = int(time.time())
        benchmark_path = f"{self.output_dir}/ann_benchmark_{scenario_name}_bs{batch_size}_{timestamp}.txt"
        
        with open(benchmark_path, "w") as f:
            f.write(f"=== ANN Throughput Benchmark ===\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Configuration:\n")
            f.write(f"  Batch Size: {batch_size}\n")
            f.write(f"  Device: {self.device}\n")
            f.write(f"  Mode: {'Inference' if batch_size == 1 else 'Training'}\n\n")
            
            f.write(f"Timing Results (ms):\n")
            f.write(f"  Average total time: {avg_total*1000:.2f}\n")
            f.write(f"  Average forward time: {avg_forward*1000:.2f}\n")
            if batch_size > 1:
                f.write(f"  Average backward time: {avg_backward*1000:.2f}\n")
            else:
                f.write(f"  Average backward time: 0.00 (inference mode)\n")
            f.write(f"\n")
            
            f.write(f"Throughput:\n")
            f.write(f"  Samples/sec: {throughput:.2f}\n")
            f.write(f"  Steps/sec: {1/avg_total:.2f}\n\n")
            
            # Additional metrics
            f.write(f"Performance Ratios:\n")
            f.write(f"  Forward/Total: {avg_forward/avg_total*100:.1f}%\n")
            if batch_size > 1:
                f.write(f"  Backward/Total: {avg_backward/avg_total*100:.1f}%\n")
            else:
                f.write(f"  Backward/Total: 0.0% (inference mode)\n")
        
        print(f"   📁 Benchmark results saved: {benchmark_path}")
        
        # Print summary
        print(f"\n⚡ Throughput Summary ({scenario_name}):")
        print(f"   Total time: {avg_total*1000:.2f} ms/step")
        print(f"   Forward time: {avg_forward*1000:.2f} ms ({avg_forward/avg_total*100:.1f}%)")
        if batch_size > 1:
            print(f"   Backward time: {avg_backward*1000:.2f} ms ({avg_backward/avg_total*100:.1f}%)")
        else:
            print(f"   Backward time: 0.00 ms (inference mode)")
        print(f"   Throughput: {throughput:.2f} samples/sec")
        
        return {
            'scenario': scenario_name,
            'batch_size': batch_size,
            'total_time': avg_total,
            'forward_time': avg_forward,
            'backward_time': avg_backward,
            'throughput': throughput
        }
    
    def run_scenario(self, scenario_name, batch_size, steps, warmup_steps, methods):
        """Run profiling for a specific scenario"""
        
        print(f"\n{'='*60}")
        print(f"🎯 SCENARIO: {scenario_name.upper()}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Batch Size: {batch_size}")
        print(f"  Steps: {steps}")
        print(f"  Warmup: {warmup_steps}")
        print(f"  Methods: {', '.join(methods)}")
        
        # Create test data
        data, target = self.create_test_data(batch_size)
        
        # Build model
        model = self.build_ann_model()
        
        # Warmup
        self.warmup_model(model, data, target, warmup_steps)
        
        # Run profiling methods
        results = {}
        
        if 'torch' in methods:
            self.profile_torch(model, data, target, steps, scenario_name, batch_size)
        
        if 'memory' in methods:
            self.profile_memory(model, data, target, min(steps, 50), scenario_name, batch_size)
        
        if 'benchmark' in methods:
            results = self.benchmark_throughput(model, data, target, steps, scenario_name, batch_size)
        
        return results


def generate_comparison_report(results_list, output_dir):
    """Generate comparative analysis report"""
    
    timestamp = int(time.time())
    report_path = f"{output_dir}/ann_comparison_report_{timestamp}.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ANN PERFORMANCE COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {time.ctime()}\n\n")
        
        # Summary table
        f.write("SCENARIO COMPARISON:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Scenario':<15} {'Batch':<8} {'Total(ms)':<12} {'Forward(ms)':<12} {'Backward(ms)':<12} {'Throughput':<12}\n")
        f.write("-" * 70 + "\n")
        
        for result in results_list:
            f.write(f"{result['scenario']:<15} {result['batch_size']:<8} "
                   f"{result['total_time']*1000:<12.2f} {result['forward_time']*1000:<12.2f} "
                   f"{result['backward_time']*1000:<12.2f} {result['throughput']:<12.1f}\n")
        
        f.write("-" * 70 + "\n\n")
        
        # Detailed analysis
        f.write("ANALYSIS:\n\n")
        
        if len(results_list) >= 2:
            offline = next((r for r in results_list if r['scenario'] == 'offline'), None)
            online = next((r for r in results_list if r['scenario'] == 'online'), None)
            
            if offline and online:
                f.write("Offline vs Online Comparison:\n")
                f.write(f"  Batch processing efficiency: {offline['throughput']/online['throughput']:.2f}x\n")
                f.write(f"  Per-sample latency (online): {online['total_time']*1000:.2f}ms\n")
                f.write(f"  Per-sample latency (offline): {offline['total_time']*1000/offline['batch_size']:.2f}ms\n")
                f.write(f"  Memory efficiency (batch 32 vs 1): {offline['batch_size']/online['batch_size']:.0f}x data per iteration\n\n")
        
        # Performance characteristics
        f.write("ANN Performance Characteristics:\n")
        for result in results_list:
            forward_ratio = result['forward_time'] / result['total_time'] * 100
            backward_ratio = result['backward_time'] / result['total_time'] * 100
            f.write(f"  {result['scenario'].capitalize()} (batch={result['batch_size']}):\n")
            f.write(f"    Forward pass: {forward_ratio:.1f}% of total time\n")
            f.write(f"    Backward pass: {backward_ratio:.1f}% of total time\n")
            f.write(f"    Efficiency: {result['throughput']:.1f} samples/sec\n\n")
        
        f.write("Recommendations:\n")
        f.write("  1. Use batch processing (batch_size=32) for offline training/evaluation\n")
        f.write("  2. Use single sample processing (batch_size=1) for real-time inference\n")
        f.write("  3. Forward pass is typically 15-25% of total training time\n")
        f.write("  4. Backward pass dominates training time (75-85%)\n")
        f.write("  5. For inference-only workloads, expect much higher throughput\n")
    
    print(f"\n📄 Comparison report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='ANN Performance Profiler - Baseline for SNN Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  offline   - Batch processing (batch_size=32, 100 steps)
  online    - Real-time streaming (batch_size=1, 200 steps)  
  both      - Run both scenarios for comparison

Examples:
  python profile/scripts/ann_profiler.py --scenario offline
  python profile/scripts/ann_profiler.py --scenario online  
  python profile/scripts/ann_profiler.py --scenario both --methods benchmark
        """
    )
    
    parser.add_argument('--scenario', choices=['offline', 'online', 'both'], 
                       default='both', help='Profiling scenario to run')
    parser.add_argument('--methods', nargs='+', 
                       choices=['torch', 'memory', 'benchmark'], 
                       default=['benchmark'], 
                       help='Profiling methods to use')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--ann-path', default='checkpoints/resnet/best_ANN.pth',
                       help='Path to ANN model')
    
    args = parser.parse_args()
    
    print("🚀 ANN Performance Profiler")
    print("=" * 50)
    print("Purpose: Baseline comparison for SNN performance analysis")
    
    # Initialize profiler
    profiler = ANNProfiler(device=args.device)
    
    # Define scenarios
    scenarios = {
        'offline': {
            'batch_size': 32,
            'steps': 100,
            'warmup': 20,
            'description': 'Batch processing simulation'
        },
        'online': {
            'batch_size': 1,
            'steps': 200,
            'warmup': 20,
            'description': 'Real-time streaming simulation'
        }
    }
    
    # Run scenarios
    results_list = []
    
    if args.scenario == 'both':
        scenarios_to_run = ['offline', 'online']
    else:
        scenarios_to_run = [args.scenario]
    
    for scenario_name in scenarios_to_run:
        scenario = scenarios[scenario_name]
        print(f"\n🎯 Running {scenario_name} scenario ({scenario['description']})")
        
        result = profiler.run_scenario(
            scenario_name=scenario_name,
            batch_size=scenario['batch_size'],
            steps=scenario['steps'],
            warmup_steps=scenario['warmup'],
            methods=args.methods
        )
        
        if result:
            results_list.append(result)
    
    # Generate comparison report if multiple scenarios
    if len(results_list) > 1:
        generate_comparison_report(results_list, profiler.output_dir)
    
    print(f"\n✅ ANN profiling completed!")
    print(f"📁 Results saved in: profile/outputs/")
    print(f"📊 View Chrome trace files at: chrome://tracing/")
    
    if results_list:
        print(f"\n📈 Summary:")
        for result in results_list:
            print(f"  {result['scenario'].capitalize()}: {result['throughput']:.1f} samples/sec "
                  f"(batch={result['batch_size']}, {result['total_time']*1000:.1f}ms/step)")


if __name__ == "__main__":
    main()