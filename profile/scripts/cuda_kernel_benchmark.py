#!/usr/bin/env python3
"""
ST-BIF CUDA内核基准测试
======================

专门针对原始CUDA内核的性能基准测试
测试不同输入规模下的性能表现
"""

import sys
import os
import time
import json
import warnings
from datetime import datetime
from typing import Dict, List

import torch
import torch.cuda.nvtx as nvtx
import numpy as np

# Add project root to path
sys.path.append('../..')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

# Import components
from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS


class ST_BIFKernelBenchmark:
    """ST-BIF CUDA内核基准测试器"""
    
    def __init__(self, num_runs: int = 100):
        self.num_runs = num_runs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu':
            raise RuntimeError("需要CUDA设备进行基准测试")
        
        self.results = {
            'config': {
                'num_runs': num_runs,
                'device': torch.cuda.get_device_name(0),
                'timestamp': datetime.now().isoformat()
            },
            'benchmarks': {}
        }
        
    def create_test_neuron(self, time_steps: int = 8) -> ST_BIFNeuron_MS:
        """创建测试用的ST-BIF神经元"""
        neuron = ST_BIFNeuron_MS(
            q_threshold=torch.tensor(1.0),
            level=8,
            sym=True,
            first_neuron=True
        ).to(self.device)
        
        neuron.T = time_steps  # 设置时间步数
        neuron.eval()
        return neuron
    
    def generate_test_data(self, batch_size: int, time_steps: int, feature_size: int) -> torch.Tensor:
        """生成测试数据"""
        torch.manual_seed(42)  # 确保可重现
        return torch.randn(time_steps * batch_size, feature_size, device=self.device)
    
    def benchmark_configuration(self, batch_size: int, time_steps: int, feature_size: int) -> Dict:
        """对指定配置进行基准测试"""
        config_name = f"B{batch_size}_T{time_steps}_F{feature_size}"
        print(f"基准测试: {config_name}")
        
        # 创建神经元和数据
        neuron = self.create_test_neuron(time_steps)
        data = self.generate_test_data(batch_size, time_steps, feature_size)
        
        # 预热
        print(f"  预热中...")
        with torch.no_grad():
            for _ in range(10):
                neuron.reset()
                _ = neuron(data)
        
        torch.cuda.synchronize()
        
        # 基准测试
        print(f"  运行 {self.num_runs} 次测试...")
        run_times = []
        memory_stats = []
        
        for run_idx in range(self.num_runs):
            # 内存测量
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
            neuron.reset()
            
            # 使用NVTX标记
            nvtx.range_push(f"ST_BIF_{config_name}_Run_{run_idx}")
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = neuron(data)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            nvtx.range_pop()
            
            # 记录时间
            run_time = (end_time - start_time) * 1000  # 转换为ms
            run_times.append(run_time)
            
            # 记录内存
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            
            memory_stats.append({
                'start_mb': start_mem / 1024**2,
                'end_mb': end_mem / 1024**2,
                'peak_mb': peak_mem / 1024**2,
                'used_mb': (end_mem - start_mem) / 1024**2
            })
        
        # 计算统计数据
        run_times = np.array(run_times)
        total_samples = batch_size * time_steps
        throughput = total_samples / (np.mean(run_times) / 1000)
        
        results = {
            'config': {
                'batch_size': batch_size,
                'time_steps': time_steps,
                'feature_size': feature_size,
                'total_samples': total_samples
            },
            'timing': {
                'mean_ms': float(np.mean(run_times)),
                'std_ms': float(np.std(run_times)),
                'min_ms': float(np.min(run_times)),
                'max_ms': float(np.max(run_times)),
                'median_ms': float(np.median(run_times)),
                'p95_ms': float(np.percentile(run_times, 95)),
                'p99_ms': float(np.percentile(run_times, 99))
            },
            'throughput': {
                'samples_per_sec': float(throughput),
                'samples_per_ms': float(total_samples / np.mean(run_times)),
                'ms_per_sample': float(np.mean(run_times) / total_samples)
            },
            'memory': {
                'mean_peak_mb': float(np.mean([m['peak_mb'] for m in memory_stats])),
                'max_peak_mb': float(np.max([m['peak_mb'] for m in memory_stats])),
                'mean_used_mb': float(np.mean([m['used_mb'] for m in memory_stats]))
            },
            'raw_times': run_times.tolist()
        }
        
        print(f"  结果: {results['timing']['mean_ms']:.3f}±{results['timing']['std_ms']:.3f} ms")
        print(f"  吞吐量: {results['throughput']['samples_per_sec']:.0f} samples/sec")
        print(f"  峰值内存: {results['memory']['mean_peak_mb']:.1f} MB")
        
        return results
    
    def run_comprehensive_benchmark(self) -> None:
        """运行综合基准测试"""
        print("ST-BIF CUDA内核综合基准测试")
        print("=" * 50)
        print(f"设备: {torch.cuda.get_device_name(0)}")
        print(f"测试轮数: {self.num_runs}")
        print()
        
        # 定义测试配置
        test_configs = [
            # (batch_size, time_steps, feature_size)
            (16, 4, 256),    # 小规模
            (32, 8, 256),    # 中等规模
            (32, 8, 512),    # 标准配置
            (64, 8, 512),    # 大批次
            (32, 16, 512),   # 长时间步
            (32, 8, 1024),   # 大特征
            (64, 16, 1024),  # 大规模
        ]
        
        nvtx.range_push("ST_BIF_Comprehensive_Benchmark")
        
        for batch_size, time_steps, feature_size in test_configs:
            config_name = f"B{batch_size}_T{time_steps}_F{feature_size}"
            
            try:
                results = self.benchmark_configuration(batch_size, time_steps, feature_size)
                self.results['benchmarks'][config_name] = results
            except Exception as e:
                print(f"  配置 {config_name} 测试失败: {str(e)}")
                self.results['benchmarks'][config_name] = {'error': str(e)}
            
            print()
        
        nvtx.range_pop()
        
        # 保存结果
        self.save_results()
        
        # 打印摘要
        self.print_summary()
    
    def save_results(self) -> None:
        """保存结果到文件"""
        os.makedirs("../outputs/nsys_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细JSON结果
        json_path = f"../outputs/nsys_results/st_bif_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存人类可读摘要
        summary_path = f"../outputs/nsys_results/st_bif_benchmark_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("ST-BIF CUDA内核基准测试结果\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"测试配置:\n")
            f.write(f"  设备: {self.results['config']['device']}\n")
            f.write(f"  测试轮数: {self.results['config']['num_runs']}\n")
            f.write(f"  时间戳: {self.results['config']['timestamp']}\n\n")
            
            f.write("基准测试结果:\n")
            f.write("-" * 20 + "\n")
            
            for config_name, results in self.results['benchmarks'].items():
                if 'error' in results:
                    f.write(f"{config_name}: 错误 - {results['error']}\n")
                    continue
                
                f.write(f"\n{config_name}:\n")
                f.write(f"  配置: B={results['config']['batch_size']}, T={results['config']['time_steps']}, F={results['config']['feature_size']}\n")
                f.write(f"  平均时间: {results['timing']['mean_ms']:.3f} ± {results['timing']['std_ms']:.3f} ms\n")
                f.write(f"  中位数时间: {results['timing']['median_ms']:.3f} ms\n")
                f.write(f"  P95时间: {results['timing']['p95_ms']:.3f} ms\n")
                f.write(f"  吞吐量: {results['throughput']['samples_per_sec']:.0f} samples/sec\n")
                f.write(f"  每样本时间: {results['throughput']['ms_per_sample']:.6f} ms\n")
                f.write(f"  峰值内存: {results['memory']['mean_peak_mb']:.1f} MB\n")
        
        print(f"结果已保存:")
        print(f"  详细数据: {json_path}")
        print(f"  摘要报告: {summary_path}")
    
    def print_summary(self) -> None:
        """打印性能摘要"""
        print("性能摘要")
        print("=" * 30)
        
        # 按吞吐量排序
        configs_by_throughput = []
        for config_name, results in self.results['benchmarks'].items():
            if 'error' not in results:
                configs_by_throughput.append((
                    config_name,
                    results['throughput']['samples_per_sec'],
                    results['timing']['mean_ms'],
                    results['memory']['mean_peak_mb']
                ))
        
        configs_by_throughput.sort(key=lambda x: x[1], reverse=True)
        
        print("\n按吞吐量排序 (samples/sec):")
        print("配置            吞吐量        平均时间    峰值内存")
        print("-" * 55)
        
        for config, throughput, time_ms, memory_mb in configs_by_throughput:
            print(f"{config:<15} {throughput:>8.0f}    {time_ms:>7.3f}ms   {memory_mb:>6.1f}MB")
        
        # 找出最佳配置
        if configs_by_throughput:
            best_config = configs_by_throughput[0]
            print(f"\n最佳性能配置: {best_config[0]}")
            print(f"  吞吐量: {best_config[1]:.0f} samples/sec")
            print(f"  平均时间: {best_config[2]:.3f} ms")


def main():
    """主函数"""
    print("ST-BIF CUDA内核基准测试器")
    print("=" * 40)
    
    # 创建基准测试器
    benchmark = ST_BIFKernelBenchmark(num_runs=50)
    
    # 运行综合基准测试
    benchmark.run_comprehensive_benchmark()
    
    print("\n基准测试完成!")


if __name__ == "__main__":
    main()