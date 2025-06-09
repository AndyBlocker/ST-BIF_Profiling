#!/usr/bin/env python3
"""
CUDA Equivalence Guard
专门保护CUDA内核等效性的CI组件，防止新版内核引入数值错误
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class CUDAEquivalenceGuard:
    def __init__(self, tolerance_config=None, verbose=True):
        self.verbose = verbose
        self.tolerance_config = tolerance_config or {
            'fp64': {'atol': 1e-7, 'rtol': 1e-5},
            'fp32': {'atol': 1e-5, 'rtol': 1e-4}, 
            'fp16': {'atol': 1e-3, 'rtol': 1e-2}
        }
        
        # 测试配置
        self.test_configs = [
            {'batch_size': 16, 'feature_dims': [64, 128, 256], 'time_steps': [4, 8]},
            {'batch_size': 32, 'feature_dims': [32, 64, 128], 'time_steps': [8, 16]},
            {'batch_size': 8, 'feature_dims': [512], 'time_steps': [4]}  # 大特征维度测试
        ]
        
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': [],
            'summary': {}
        }
    
    def log(self, message, level='INFO'):
        if self.verbose:
            print(f"[{level}] {message}")
    
    def load_kernels(self):
        """加载新旧CUDA内核"""
        try:
            # 加载原始稳定内核
            sys.path.insert(0, str(project_root / "neuron_cupy"))
            from cuda_operator import ST_BIFNodeATGF_MS_CUDA as OriginalKernel
            
            # 尝试加载新版内核
            try:
                from cuda_operator_new import ST_BIFNodeATGF_MS_CUDA as NewKernel
                has_new_kernel = True
                self.log("成功加载新旧CUDA内核")
            except ImportError as e:
                self.log(f"无法加载新版CUDA内核: {e}", "WARN")
                has_new_kernel = False
                NewKernel = None
            
            return OriginalKernel, NewKernel, has_new_kernel
            
        except ImportError as e:
            self.log(f"无法加载CUDA内核: {e}", "ERROR")
            raise
    
    def generate_test_data(self, batch_size, feature_dim, time_step, dtype=torch.float32):
        """生成测试数据"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 生成输入数据
        x = torch.randn(batch_size, time_step, feature_dim, dtype=dtype, device=device)
        threshold = torch.ones(feature_dim, dtype=dtype, device=device) * 0.5
        
        # 确保数据有一定的动态范围
        x = x * 2.0  # 扩大动态范围
        
        return x, threshold
    
    def run_kernel_test(self, kernel_class, x, threshold, level=8, T_max=1.0, decay=0.0):
        """运行单个内核测试"""
        try:
            # 重置CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 运行内核
            start_time = time.time()
            
            # 确保T_max是tensor
            if isinstance(T_max, (int, float)):
                T_max = torch.tensor(T_max, dtype=x.dtype, device=x.device)
            
            y, mem_potential, spike_count = kernel_class.apply(x, threshold, level, T_max, decay)
            
            end_time = time.time()
            
            return {
                'success': True,
                'output': y,
                'mem_potential': mem_potential, 
                'spike_count': spike_count,
                'execution_time': end_time - start_time,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'memory_allocated': 0
            }
    
    def compare_results(self, result1, result2, precision='fp32'):
        """比较两个内核的输出结果"""
        if not (result1['success'] and result2['success']):
            return {
                'equivalent': False,
                'reason': 'One or both kernels failed',
                'result1_success': result1['success'],
                'result2_success': result2['success']
            }
        
        tolerance = self.tolerance_config[precision]
        
        try:
            # 比较主要输出
            output_close = torch.allclose(
                result1['output'], result2['output'],
                atol=tolerance['atol'], rtol=tolerance['rtol']
            )
            
            # 比较膜电位
            mem_close = torch.allclose(
                result1['mem_potential'], result2['mem_potential'],
                atol=tolerance['atol'], rtol=tolerance['rtol']
            )
            
            # 比较脉冲计数
            spike_close = torch.allclose(
                result1['spike_count'], result2['spike_count'],
                atol=tolerance['atol'], rtol=tolerance['rtol']
            )
            
            equivalent = output_close and mem_close and spike_close
            
            # 计算实际差异
            output_diff = torch.max(torch.abs(result1['output'] - result2['output'])).item()
            mem_diff = torch.max(torch.abs(result1['mem_potential'] - result2['mem_potential'])).item()
            spike_diff = torch.max(torch.abs(result1['spike_count'] - result2['spike_count'])).item()
            
            return {
                'equivalent': equivalent,
                'output_close': output_close,
                'mem_close': mem_close, 
                'spike_close': spike_close,
                'max_output_diff': output_diff,
                'max_mem_diff': mem_diff,
                'max_spike_diff': spike_diff,
                'tolerance_used': tolerance,
                'performance_ratio': result2['execution_time'] / result1['execution_time'] if result1['execution_time'] > 0 else float('inf')
            }
            
        except Exception as e:
            return {
                'equivalent': False,
                'reason': f'Comparison failed: {str(e)}'
            }
    
    def run_equivalence_tests(self):
        """运行完整的等效性测试套件"""
        if not torch.cuda.is_available():
            self.log("CUDA不可用，跳过等效性测试", "WARN")
            return False
        
        # 加载内核
        original_kernel, new_kernel, has_new_kernel = self.load_kernels()
        
        if not has_new_kernel:
            self.log("新版内核不可用，只测试原始内核", "WARN")
            return self.test_single_kernel(original_kernel)
        
        self.log("开始CUDA内核等效性测试...")
        
        all_passed = True
        
        # 测试不同精度
        for precision in ['fp32', 'fp16']:  # 暂时跳过fp64，因为某些GPU不支持
            dtype = torch.float32 if precision == 'fp32' else torch.float16
            
            self.log(f"测试精度: {precision}")
            
            # 测试不同配置
            for config in self.test_configs:
                batch_size = config['batch_size']
                
                for feature_dim in config['feature_dims']:
                    for time_step in config['time_steps']:
                        test_name = f"{precision}_B{batch_size}_F{feature_dim}_T{time_step}"
                        
                        try:
                            # 生成测试数据
                            x, threshold = self.generate_test_data(batch_size, feature_dim, time_step, dtype)
                            
                            # 运行两个内核
                            result_orig = self.run_kernel_test(original_kernel, x, threshold)
                            result_new = self.run_kernel_test(new_kernel, x, threshold)
                            
                            # 比较结果
                            comparison = self.compare_results(result_orig, result_new, precision)
                            
                            # 记录测试结果
                            test_result = {
                                'test_name': test_name,
                                'config': {
                                    'precision': precision,
                                    'batch_size': batch_size,
                                    'feature_dim': feature_dim,
                                    'time_step': time_step
                                },
                                'original_kernel': result_orig,
                                'new_kernel': result_new,
                                'comparison': comparison,
                                'passed': comparison['equivalent']
                            }
                            
                            self.results['test_details'].append(test_result)
                            self.results['total_tests'] += 1
                            
                            if comparison['equivalent']:
                                self.results['passed_tests'] += 1
                                self.log(f"✓ {test_name}: 等效性验证通过 (性能比率: {comparison.get('performance_ratio', 'N/A'):.3f})")
                            else:
                                self.results['failed_tests'] += 1
                                all_passed = False
                                self.log(f"✗ {test_name}: 等效性验证失败", "ERROR")
                                self.log(f"  最大输出差异: {comparison.get('max_output_diff', 'N/A')}", "ERROR")
                                self.log(f"  容差: {comparison.get('tolerance_used', {})}", "ERROR")
                                
                                # 对于失败的测试，提供更多信息
                                if not comparison.get('output_close', True):
                                    self.log("  主要输出不匹配", "ERROR")
                                if not comparison.get('mem_close', True):
                                    self.log("  膜电位不匹配", "ERROR")
                                if not comparison.get('spike_close', True):
                                    self.log("  脉冲计数不匹配", "ERROR")
                            
                        except Exception as e:
                            self.results['failed_tests'] += 1
                            self.results['total_tests'] += 1
                            all_passed = False
                            self.log(f"✗ {test_name}: 测试异常: {e}", "ERROR")
        
        # 生成摘要
        self.results['summary'] = {
            'all_tests_passed': all_passed,
            'pass_rate': self.results['passed_tests'] / self.results['total_tests'] if self.results['total_tests'] > 0 else 0,
            'total_configs_tested': len(self.test_configs) * sum(len(c['feature_dims']) * len(c['time_steps']) for c in self.test_configs) * 2,  # 2个精度
            'has_new_kernel': has_new_kernel
        }
        
        return all_passed
    
    def test_single_kernel(self, kernel_class):
        """测试单个内核的基本功能"""
        self.log("运行单内核功能测试...")
        
        try:
            # 简单功能测试
            x, threshold = self.generate_test_data(16, 64, 8)
            result = self.run_kernel_test(kernel_class, x, threshold)
            
            if result['success']:
                self.log("✓ 单内核功能测试通过")
                return True
            else:
                self.log(f"✗ 单内核功能测试失败: {result['error']}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"✗ 单内核测试异常: {e}", "ERROR")
            return False
    
    def save_results(self, output_file):
        """保存测试结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"等效性测试结果已保存到: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="CUDA内核等效性保护测试")
    parser.add_argument("--output", "-o", default="ci/results/latest/cuda_equivalence_test.json",
                       help="输出文件路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    parser.add_argument("--tolerance-config", help="自定义容差配置文件")
    
    args = parser.parse_args()
    
    # 加载自定义容差配置
    tolerance_config = None
    if args.tolerance_config and os.path.exists(args.tolerance_config):
        with open(args.tolerance_config) as f:
            tolerance_config = json.load(f)
    
    # 创建等效性守护器
    guard = CUDAEquivalenceGuard(tolerance_config, verbose=not args.quiet)
    
    try:
        # 运行等效性测试
        success = guard.run_equivalence_tests()
        
        # 保存结果
        guard.save_results(args.output)
        
        # 输出摘要
        summary = guard.results['summary']
        if not args.quiet:
            print("\n" + "="*50)
            print("CUDA内核等效性测试摘要")
            print("="*50)
            print(f"总测试数: {guard.results['total_tests']}")
            print(f"通过测试: {guard.results['passed_tests']}")
            print(f"失败测试: {guard.results['failed_tests']}")
            print(f"通过率: {summary['pass_rate']:.1%}")
            print(f"整体结果: {'✓ 通过' if summary['all_tests_passed'] else '✗ 失败'}")
        
        # 设置退出码
        sys.exit(0 if success else 1)
        
    except Exception as e:
        guard.log(f"等效性测试发生错误: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()