#!/usr/bin/env python3
"""
SNN Inference NVTX Profiling Script

This script creates detailed NVTX markers for nsys profiling of SNN inference.
Designed for batch_size=32 with 3-5 inference runs.
Saves key metrics to text files with nsys-compatible naming.
"""

import sys
import os
import time
import json
from datetime import datetime
import torch
import torch.cuda.nvtx as nvtx
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.resnet import resnet18
from snn.conversion.quantization import myquan_replace_resnet
from wrapper.snn_wrapper import SNNWrapper_MS
from wrapper.encoding import get_subtensors

class NVTXSNNProfiler:
    def __init__(self, batch_size=32, num_runs=5):
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {
            'run_times': [],
            'memory_stats': [],
            'component_times': {},
            'metadata': {
                'batch_size': batch_size,
                'num_runs': num_runs,
                'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
                'timestamp': datetime.now().isoformat()
            }
        }
        
    def load_snn_model(self):
        """Load SNN model with NVTX markers"""
        nvtx.range_push("Model_Loading")
        
        print(f"Loading SNN model for profiling...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of runs: {self.num_runs}")
        
        # Load QANN first
        nvtx.range_push("QANN_Creation")
        qann_model = resnet18(num_classes=10).to(self.device)
        myquan_replace_resnet(qann_model, level=8, weight_bit=32)
        qann_checkpoint = torch.load('/home/zilingwei/Projects/ST-BIF_Profiling/checkpoints/resnet/best_QANN.pth')
        qann_model.load_state_dict(qann_checkpoint)
        qann_model.eval()
        nvtx.range_pop()
        
        # Create SNN
        nvtx.range_push("SNN_Wrapper_Creation")
        self.snn_model = SNNWrapper_MS(
            ann_model=qann_model,
            time_step=8,
            level=8,
            neuron_type="ST-BIF"
        ).to(self.device)
        self.snn_model.eval()
        nvtx.range_pop()
        
        nvtx.range_pop()  # Model_Loading
        print("✓ SNN model loaded successfully")
        
    def create_input_data(self):
        """Create input data with NVTX markers"""
        nvtx.range_push("Input_Data_Creation")
        
        # Create random input data (CIFAR-10 format)
        input_data = torch.randn(self.batch_size, 3, 32, 32, device=self.device)
        
        # Normalize to typical CIFAR-10 range
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=self.device).view(1, 3, 1, 1)
        input_data = (input_data * std) + mean
        
        nvtx.range_pop()
        return input_data
        
    def detailed_snn_forward_with_nvtx(self, input_data, run_idx):
        """Detailed SNN forward pass with comprehensive NVTX markers"""
        nvtx.range_push(f"SNN_Inference_Run_{run_idx}")
        
        component_times = {}
        
        # 1. Reset model state
        nvtx.range_push("Model_Reset")
        start_time = time.perf_counter()
        self.snn_model._reset_all_states()
        torch.cuda.synchronize()
        component_times['reset'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # 2. Time encoding
        nvtx.range_push("Time_Encoding")
        start_time = time.perf_counter()
        input_seq = get_subtensors(
            input_data, 0.0, 0.0, 
            sample_grain=self.snn_model.step, 
            time_step=self.snn_model.T
        )
        torch.cuda.synchronize()
        component_times['encoding'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # 3. Input reshaping
        nvtx.range_push("Input_Reshaping")
        start_time = time.perf_counter()
        T, B, C, H, W = input_seq.shape
        input_reshaped = input_seq.reshape(T*B, C, H, W)
        torch.cuda.synchronize()
        component_times['reshaping'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # 4. Model forward pass (the main computation)
        nvtx.range_push("Model_Forward_Pass")
        start_time = time.perf_counter()
        
        # Break down model forward into major components
        nvtx.range_push("Model_Compute_All_Layers")
        output = self.snn_model.model(input_reshaped)
        torch.cuda.synchronize()
        nvtx.range_pop()
        
        component_times['model_forward'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # 5. Output processing
        nvtx.range_push("Output_Processing")
        start_time = time.perf_counter()
        output_reshaped = output.reshape(torch.Size([T, B]) + output.shape[1:])
        final_output = output_reshaped.sum(dim=0)
        torch.cuda.synchronize()
        component_times['output_processing'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        nvtx.range_pop()  # SNN_Inference_Run_{run_idx}
        
        return final_output, component_times
        
    def detailed_layer_profiling(self, input_data, run_idx):
        """Profile individual layers with NVTX markers"""
        nvtx.range_push(f"Detailed_Layer_Profiling_Run_{run_idx}")
        
        # Reset and encode
        self.snn_model._reset_all_states()
        input_seq = get_subtensors(
            input_data, 0.0, 0.0, 
            sample_grain=self.snn_model.step, 
            time_step=self.snn_model.T
        )
        T, B, C, H, W = input_seq.shape
        input_reshaped = input_seq.reshape(T*B, C, H, W)
        
        # Profile major layer groups
        layer_times = {}
        x = input_reshaped
        
        # Conv1 + BN1 + ReLU
        nvtx.range_push("Conv1_Block")
        start_time = time.perf_counter()
        x = self.snn_model.model.conv1(x)
        x = self.snn_model.model.bn1(x)
        x = self.snn_model.model.relu(x)
        torch.cuda.synchronize()
        layer_times['conv1_block'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # MaxPool
        nvtx.range_push("MaxPool")
        start_time = time.perf_counter()
        x = self.snn_model.model.maxpool(x)
        torch.cuda.synchronize()
        layer_times['maxpool'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # Layer1 (ResNet blocks)
        nvtx.range_push("Layer1_ResBlocks")
        start_time = time.perf_counter()
        x = self.snn_model.model.layer1(x)
        torch.cuda.synchronize()
        layer_times['layer1'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # Layer2
        nvtx.range_push("Layer2_ResBlocks")
        start_time = time.perf_counter()
        x = self.snn_model.model.layer2(x)
        torch.cuda.synchronize()
        layer_times['layer2'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # Layer3
        nvtx.range_push("Layer3_ResBlocks")
        start_time = time.perf_counter()
        x = self.snn_model.model.layer3(x)
        torch.cuda.synchronize()
        layer_times['layer3'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # Layer4
        nvtx.range_push("Layer4_ResBlocks")
        start_time = time.perf_counter()
        x = self.snn_model.model.layer4(x)
        torch.cuda.synchronize()
        layer_times['layer4'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        # AvgPool + FC
        nvtx.range_push("Final_Layers")
        start_time = time.perf_counter()
        x = self.snn_model.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.snn_model.model.fc(x)
        torch.cuda.synchronize()
        layer_times['final_layers'] = (time.perf_counter() - start_time) * 1000
        nvtx.range_pop()
        
        nvtx.range_pop()  # Detailed_Layer_Profiling_Run_{run_idx}
        
        return layer_times
        
    def memory_profiling(self, run_idx):
        """Profile memory usage with NVTX markers"""
        nvtx.range_push(f"Memory_Profiling_Run_{run_idx}")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_memory = torch.cuda.memory_allocated()
            
            # Run inference
            input_data = self.create_input_data()
            with torch.no_grad():
                output = self.snn_model(input_data)
            
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_stats = {
                'start_memory_mb': start_memory / 1024**2,
                'end_memory_mb': end_memory / 1024**2,
                'peak_memory_mb': peak_memory / 1024**2,
                'memory_increase_mb': (end_memory - start_memory) / 1024**2
            }
        else:
            memory_stats = {'error': 'CUDA not available'}
            
        nvtx.range_pop()
        return memory_stats
        
    def run_profiling_session(self):
        """Run complete profiling session with NVTX markers"""
        nvtx.range_push("Complete_Profiling_Session")
        
        print(f"\n{'='*60}")
        print(f"Starting NVTX Profiling Session")
        print(f"{'='*60}")
        
        # Load model
        self.load_snn_model()
        
        # Warmup runs
        nvtx.range_push("Warmup_Runs")
        print("Running warmup iterations...")
        input_data = self.create_input_data()
        with torch.no_grad():
            for i in range(2):
                nvtx.range_push(f"Warmup_{i}")
                _ = self.snn_model(input_data)
                nvtx.range_pop()
        nvtx.range_pop()
        
        # Main profiling runs
        print(f"\nRunning {self.num_runs} profiled inference iterations...")
        
        for run_idx in range(self.num_runs):
            print(f"  Run {run_idx + 1}/{self.num_runs}")
            
            # Create fresh input for each run
            input_data = self.create_input_data()
            
            # Main inference timing
            nvtx.range_push(f"Main_Inference_Timing_Run_{run_idx}")
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output, component_times = self.detailed_snn_forward_with_nvtx(input_data, run_idx)
                
            torch.cuda.synchronize()
            total_time = (time.perf_counter() - start_time) * 1000
            nvtx.range_pop()
            
            # Store results
            self.results['run_times'].append(total_time)
            for comp, time_val in component_times.items():
                if comp not in self.results['component_times']:
                    self.results['component_times'][comp] = []
                self.results['component_times'][comp].append(time_val)
            
            # Memory profiling
            memory_stats = self.memory_profiling(run_idx)
            self.results['memory_stats'].append(memory_stats)
            
            # Detailed layer profiling (only for first few runs to avoid overhead)
            if run_idx < 2:
                layer_times = self.detailed_layer_profiling(input_data, run_idx)
                if 'layer_times' not in self.results:
                    self.results['layer_times'] = {}
                for layer, time_val in layer_times.items():
                    if layer not in self.results['layer_times']:
                        self.results['layer_times'][layer] = []
                    self.results['layer_times'][layer].append(time_val)
        
        nvtx.range_pop()  # Complete_Profiling_Session
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Save results
        self.save_results()
        
        print("\n✓ Profiling session completed!")
        print(f"✓ Results saved to nsys_profiling_results.json")
        print(f"✓ Summary saved to nsys_profiling_summary.txt")
        
    def calculate_statistics(self):
        """Calculate summary statistics"""
        run_times = np.array(self.results['run_times'])
        
        self.results['statistics'] = {
            'total_time_stats': {
                'mean_ms': float(np.mean(run_times)),
                'std_ms': float(np.std(run_times)),
                'min_ms': float(np.min(run_times)),
                'max_ms': float(np.max(run_times)),
                'throughput_samples_per_sec': float(self.batch_size / (np.mean(run_times) / 1000))
            }
        }
        
        # Component statistics
        for comp, times in self.results['component_times'].items():
            times_array = np.array(times)
            self.results['statistics'][f'{comp}_stats'] = {
                'mean_ms': float(np.mean(times_array)),
                'std_ms': float(np.std(times_array)),
                'percentage_of_total': float(np.mean(times_array) / np.mean(run_times) * 100)
            }
            
        # Memory statistics
        if self.results['memory_stats'] and 'error' not in self.results['memory_stats'][0]:
            peak_memories = [stats['peak_memory_mb'] for stats in self.results['memory_stats']]
            self.results['statistics']['memory_stats'] = {
                'mean_peak_memory_mb': float(np.mean(peak_memories)),
                'max_peak_memory_mb': float(np.max(peak_memories))
            }
        
    def save_results(self):
        """Save results to files with nsys-compatible naming"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_filename = f"nsys_profiling_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save human-readable summary
        summary_filename = f"nsys_profiling_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("ST-BIF SNN NVTX Profiling Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Batch size: {self.batch_size}\n")
            f.write(f"  Number of runs: {self.num_runs}\n")
            f.write(f"  Device: {self.results['metadata']['device']}\n")
            f.write(f"  Timestamp: {self.results['metadata']['timestamp']}\n\n")
            
            stats = self.results['statistics']
            f.write(f"Overall Performance:\n")
            f.write(f"  Mean inference time: {stats['total_time_stats']['mean_ms']:.3f} +- {stats['total_time_stats']['std_ms']:.3f} ms\n")
            f.write(f"  Throughput: {stats['total_time_stats']['throughput_samples_per_sec']:.1f} samples/sec\n")
            f.write(f"  Min/Max time: {stats['total_time_stats']['min_ms']:.3f} / {stats['total_time_stats']['max_ms']:.3f} ms\n\n")
            
            f.write(f"Component Breakdown:\n")
            for comp in ['reset', 'encoding', 'reshaping', 'model_forward', 'output_processing']:
                if f'{comp}_stats' in stats:
                    comp_stats = stats[f'{comp}_stats']
                    f.write(f"  {comp.replace('_', ' ').title()}: {comp_stats['mean_ms']:.3f} ms ({comp_stats['percentage_of_total']:.1f}%)\n")
            
            if 'memory_stats' in stats:
                f.write(f"\nMemory Usage:\n")
                f.write(f"  Mean peak memory: {stats['memory_stats']['mean_peak_memory_mb']:.1f} MB\n")
                f.write(f"  Max peak memory: {stats['memory_stats']['max_peak_memory_mb']:.1f} MB\n")
            
            # Layer timings if available
            if 'layer_times' in self.results:
                f.write(f"\nLayer Breakdown (first 2 runs):\n")
                for layer, times in self.results['layer_times'].items():
                    mean_time = np.mean(times)
                    f.write(f"  {layer.replace('_', ' ').title()}: {mean_time:.3f} ms\n")
        
        print(f"Results saved:")
        print(f"  - Detailed data: {json_filename}")
        print(f"  - Summary: {summary_filename}")

def main():
    """Main profiling function"""
    print("ST-BIF SNN NVTX Profiling Script")
    print("This script generates nsys-compatible profiling data")
    print("\nTo use with nsys, run:")
    print("nsys profile -o snn_profile python nsys_snn_profiling.py")
    print("\nOr run standalone for timing data:")
    
    # Create profiler
    profiler = NVTXSNNProfiler(batch_size=32, num_runs=5)
    
    # Run profiling
    profiler.run_profiling_session()
    
    # Print final summary
    stats = profiler.results['statistics']['total_time_stats']
    print(f"\nFinal Summary:")
    print(f"  Mean inference time: {stats['mean_ms']:.3f} +- {stats['std_ms']:.3f} ms")
    print(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Batch size: {profiler.batch_size}")

if __name__ == "__main__":
    main()