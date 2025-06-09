#!/usr/bin/env python3
"""
CUDA Kernel Benchmark - Performance testing for ST-BIF CUDA operators
Based on the original cuda_kernel_profiler.py but integrated with pytest framework
"""

import pytest
import torch
import torch.cuda.nvtx as nvtx
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

# Import kernels with error handling
try:
    from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA as OriginalKernel
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    ORIGINAL_AVAILABLE = False
    OriginalKernel = None

try:
    from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA as NewKernel
    NEW_AVAILABLE = True
except ImportError as e:
    NEW_AVAILABLE = False
    NewKernel = None


class CUDAKernelBenchmark:
    """Performance benchmark suite for CUDA kernels"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self._get_device_info(),
            'benchmarks': {}
        }
    
    def _get_device_info(self):
        """Get CUDA device information"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        device = torch.cuda.current_device()
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(device),
            'device_capability': torch.cuda.get_device_capability(device),
            'total_memory': torch.cuda.get_device_properties(device).total_memory,
            'torch_version': torch.__version__
        }
    
    def benchmark_kernel(self, kernel_class, x, threshold, level=8, T_max=1.0, decay=0.0, 
                        runs=10, warmup=3, name=""):
        """Benchmark a single kernel"""
        if not torch.cuda.is_available():
            return None
        
        # Ensure T_max is tensor
        if isinstance(T_max, (int, float)):
            T_max = torch.tensor(T_max, dtype=x.dtype, device=x.device)
        
        # Warmup
        for _ in range(warmup):
            try:
                with torch.no_grad():
                    _ = kernel_class.apply(x, threshold, level, T_max, decay)
            except Exception as e:
                return {'error': str(e), 'kernel': name}
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(runs):
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            try:
                with torch.no_grad():
                    result = kernel_class.apply(x, threshold, level, T_max, decay)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                peak_mem = torch.cuda.max_memory_allocated()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                memory_usage.append(peak_mem - start_mem)
                
                torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                return {'error': str(e), 'kernel': name}
        
        return {
            'kernel': name,
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'mean_memory_mb': np.mean(memory_usage) / 1024**2,
            'throughput_samples_per_sec': x.shape[1] * 1000 / np.mean(times),  # batch_size * 1000 / time_ms
            'successful_runs': len(times),
            'total_runs': runs
        }
    
    def compare_kernels(self, shape, dtype=torch.float32, level=8, runs=10):
        """Compare performance between different kernels"""
        device = torch.device('cuda')
        time_steps, batch_size, features = shape
        
        # Generate test data
        x = torch.randn(time_steps, batch_size, features, dtype=dtype, device=device)
        threshold = torch.ones(features, dtype=dtype, device=device) * 0.5
        
        results = {}
        
        # Benchmark original kernel
        if ORIGINAL_AVAILABLE:
            result = self.benchmark_kernel(
                OriginalKernel, x, threshold, level=level, runs=runs, name="Original"
            )
            if result:
                results['original'] = result
        
        # Benchmark new kernel
        if NEW_AVAILABLE:
            result = self.benchmark_kernel(
                NewKernel, x, threshold, level=level, runs=runs, name="New"
            )
            if result:
                results['new'] = result
        
        # Calculate comparison metrics
        if 'original' in results and 'new' in results:
            orig_time = results['original']['mean_time_ms']
            new_time = results['new']['mean_time_ms']
            results['comparison'] = {
                'speedup': orig_time / new_time,
                'time_difference_ms': new_time - orig_time,
                'time_difference_percent': ((new_time - orig_time) / orig_time) * 100
            }
        
        return results
    
    def run_comprehensive_benchmark(self, shapes=None, dtypes=None, output_dir=None):
        """Run comprehensive benchmark across multiple configurations"""
        if shapes is None:
            shapes = [
                (16, 32, 256),   # Medium
                (32, 32, 512),   # Large  
                (8, 64, 128),    # Wide batch
                (64, 16, 64),    # Long sequence
            ]
        
        if dtypes is None:
            dtypes = [torch.float32, torch.float16]
        
        if output_dir is None:
            output_dir = Path("tests/benchmark_results")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dtype in dtypes:
            dtype_name = str(dtype).split('.')[-1]  # e.g., 'float32'
            
            for shape in shapes:
                shape_name = f"T{shape[0]}_B{shape[1]}_F{shape[2]}"
                test_name = f"{dtype_name}_{shape_name}"
                
                print(f"Benchmarking {test_name}...")
                
                try:
                    results = self.compare_kernels(shape, dtype=dtype)
                    self.results['benchmarks'][test_name] = {
                        'shape': shape,
                        'dtype': dtype_name,
                        'results': results
                    }
                except Exception as e:
                    print(f"Error benchmarking {test_name}: {e}")
                    self.results['benchmarks'][test_name] = {
                        'shape': shape,
                        'dtype': dtype_name,
                        'error': str(e)
                    }
        
        # Save results
        results_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to: {results_file}")
        
        # Generate plots if matplotlib is available
        try:
            self.generate_plots(output_dir)
        except ImportError:
            print("Matplotlib not available, skipping plots")
        
        return self.results
    
    def generate_plots(self, output_dir):
        """Generate performance comparison plots"""
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        configs = []
        original_times = []
        new_times = []
        speedups = []
        
        for test_name, data in self.results['benchmarks'].items():
            if 'error' in data:
                continue
                
            results = data['results']
            if 'original' in results and 'new' in results:
                configs.append(test_name)
                original_times.append(results['original']['mean_time_ms'])
                new_times.append(results['new']['mean_time_ms'])
                if 'comparison' in results:
                    speedups.append(results['comparison']['speedup'])
                else:
                    speedups.append(1.0)
        
        if not configs:
            print("No valid comparison data for plotting")
            return
        
        # Create performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Execution times
        x = np.arange(len(configs))
        width = 0.35
        
        ax1.bar(x - width/2, original_times, width, label='Original Kernel', alpha=0.8)
        ax1.bar(x + width/2, new_times, width, label='New Kernel', alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('CUDA Kernel Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        colors = ['green' if s > 1 else 'red' for s in speedups]
        bars = ax2.bar(configs, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Speedup (Original/New)')
        ax2.set_title('Performance Speedup')
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add speedup values on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved to: {plot_file}")


# Pytest integration
class TestCUDAKernelBenchmark:
    """Pytest wrapper for CUDA kernel benchmarks"""
    
    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    def test_basic_performance_benchmark(self):
        """Basic performance benchmark test"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        if not (ORIGINAL_AVAILABLE or NEW_AVAILABLE):
            pytest.skip("No CUDA kernels available")
        
        benchmark = CUDAKernelBenchmark()
        shape = (16, 16, 128)  # Small shape for quick test
        results = benchmark.compare_kernels(shape, runs=5)
        
        # Basic assertions
        assert len(results) > 0, "No benchmark results generated"
        
        for kernel_name, result in results.items():
            if kernel_name == 'comparison':
                continue
            assert 'mean_time_ms' in result, f"Missing timing data for {kernel_name}"
            assert result['mean_time_ms'] > 0, f"Invalid timing for {kernel_name}"
            assert result['successful_runs'] > 0, f"No successful runs for {kernel_name}"
    
    @pytest.mark.cuda
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.parametrize("shape", [(16, 32, 256), (8, 64, 128)])
    def test_kernel_performance_regression(self, shape):
        """Test that new kernel doesn't regress significantly"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        if not (ORIGINAL_AVAILABLE and NEW_AVAILABLE):
            pytest.skip("Both kernels required for comparison")
        
        benchmark = CUDAKernelBenchmark()
        results = benchmark.compare_kernels(shape, runs=10)
        
        if 'comparison' in results:
            speedup = results['comparison']['speedup']
            # Allow up to 2x slowdown (new kernel shouldn't be more than 2x slower)
            assert speedup > 0.5, f"New kernel is too slow: {speedup:.2f}x speedup"
            print(f"Performance for {shape}: {speedup:.2f}x speedup")


# CLI interface
def main():
    """Main CLI interface for running benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA Kernel Benchmark")
    parser.add_argument("--output", "-o", default="tests/benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer configurations")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of benchmark runs per configuration")
    
    args = parser.parse_args()
    
    benchmark = CUDAKernelBenchmark()
    
    if args.quick:
        shapes = [(16, 32, 128)]
        dtypes = [torch.float32]
    else:
        shapes = [(16, 32, 256), (32, 32, 512), (8, 64, 128), (64, 16, 64)]
        dtypes = [torch.float32, torch.float16]
    
    print("Starting CUDA kernel benchmark...")
    results = benchmark.run_comprehensive_benchmark(
        shapes=shapes, 
        dtypes=dtypes, 
        output_dir=args.output
    )
    
    # Print summary
    print("\nBenchmark Summary:")
    for test_name, data in results['benchmarks'].items():
        if 'error' in data:
            print(f"  {test_name}: ERROR - {data['error']}")
        elif 'comparison' in data['results']:
            speedup = data['results']['comparison']['speedup']
            print(f"  {test_name}: {speedup:.2f}x speedup")
        else:
            print(f"  {test_name}: Single kernel result available")


if __name__ == "__main__":
    main()