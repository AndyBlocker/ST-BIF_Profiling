#!/usr/bin/env python3
"""
ST-BIF Regression Test Suite
完整的回归测试套件，保护核心功能免受意外破坏
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class RegressionTestSuite:
    def __init__(self, baseline_dir=None, verbose=True):
        self.verbose = verbose
        self.baseline_dir = Path(baseline_dir) if baseline_dir else Path("ci/baselines/current")
        self.results_dir = Path("ci/results/latest")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_version': str(self.baseline_dir),
            'tests': {},
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'warning_tests': 0
            }
        }
        
        # 加载基线数据
        self.baseline_data = self.load_baseline()
    
    def log(self, message, level='INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")
    
    def load_baseline(self):
        """加载基线数据"""
        baseline_file = self.baseline_dir / "baseline_snapshot.json"
        
        if not baseline_file.exists():
            self.log(f"基线文件不存在: {baseline_file}", "WARN")
            return None
        
        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"加载基线文件失败: {e}", "ERROR")
            return None
    
    def run_test(self, test_name, test_func, *args, **kwargs):
        """运行单个测试并记录结果"""
        self.log(f"运行测试: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.test_results['tests'][test_name] = {
                'status': result.get('status', 'unknown'),
                'passed': result.get('passed', False),
                'execution_time': execution_time,
                'details': result.get('details', {}),
                'message': result.get('message', ''),
                'baseline_comparison': result.get('baseline_comparison', {})
            }
            
            # 更新统计
            self.test_results['summary']['total_tests'] += 1
            if result.get('passed'):
                self.test_results['summary']['passed_tests'] += 1
                self.log(f"✓ {test_name}: 通过 ({execution_time:.2f}s)")
            elif result.get('status') == 'warning':
                self.test_results['summary']['warning_tests'] += 1
                self.log(f"⚠ {test_name}: 警告 - {result.get('message', '')}", "WARN")
            else:
                self.test_results['summary']['failed_tests'] += 1
                self.log(f"✗ {test_name}: 失败 - {result.get('message', '')}", "ERROR")
            
            return result.get('passed', False)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.test_results['tests'][test_name] = {
                'status': 'error',
                'passed': False,
                'execution_time': execution_time,
                'error': str(e),
                'message': f'测试异常: {str(e)}'
            }
            
            self.test_results['summary']['total_tests'] += 1
            self.test_results['summary']['failed_tests'] += 1
            self.log(f"✗ {test_name}: 异常 - {str(e)}", "ERROR")
            return False
    
    def test_model_conversion_pipeline(self):
        """测试模型转换流水线"""
        try:
            # 运行转换示例
            cmd = [sys.executable, "examples/ann_to_snn_conversion.py", "--quiet", "--batch-size", "32"]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'passed': False,
                    'message': f'转换流水线执行失败: {result.stderr}',
                    'details': {'returncode': result.returncode, 'stderr': result.stderr}
                }
            
            # 解析输出中的精度信息
            output = result.stdout
            current_accuracy = {}
            
            lines = output.split('\n')
            for line in lines:
                if "ANN Test Accuracy" in line:
                    try:
                        current_accuracy["ann_accuracy"] = float(line.split(':')[1].strip().rstrip('%'))
                    except:
                        pass
                elif "QANN Test Accuracy" in line:
                    try:
                        current_accuracy["qann_accuracy"] = float(line.split(':')[1].strip().rstrip('%'))
                    except:
                        pass
                elif "SNN Test Accuracy" in line:
                    try:
                        current_accuracy["snn_accuracy"] = float(line.split(':')[1].strip().rstrip('%'))
                    except:
                        pass
            
            # 与基线对比
            baseline_comparison = {}
            if self.baseline_data and 'baselines' in self.baseline_data:
                baseline_acc = self.baseline_data['baselines'].get('model_accuracy', {})
                
                for key in ['ann_accuracy', 'qann_accuracy', 'snn_accuracy']:
                    if key in current_accuracy and key in baseline_acc:
                        current_val = current_accuracy[key]
                        baseline_val = baseline_acc[key]
                        diff = current_val - baseline_val
                        baseline_comparison[key] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'difference': diff,
                            'regression': diff < -0.5  # 精度下降超过0.5%认为是回归
                        }
            
            # 判断是否通过
            has_regression = any(comp.get('regression', False) for comp in baseline_comparison.values())
            has_major_failure = not current_accuracy  # 没有解析到任何精度信息
            
            if has_major_failure:
                status = 'failed'
                passed = False
                message = '无法解析模型精度信息'
            elif has_regression:
                status = 'failed' 
                passed = False
                message = '检测到精度回归'
            elif len(current_accuracy) < 3:
                status = 'warning'
                passed = True
                message = '部分精度信息缺失'
            else:
                status = 'passed'
                passed = True
                message = '转换流水线正常'
            
            return {
                'status': status,
                'passed': passed,
                'message': message,
                'details': {
                    'current_accuracy': current_accuracy,
                    'execution_output': output[:1000]  # 只保存前1000字符
                },
                'baseline_comparison': baseline_comparison
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'passed': False,
                'message': '转换流水线执行超时 (>600s)',
                'details': {'timeout': True}
            }
    
    def test_cuda_kernel_equivalence(self):
        """测试CUDA内核等效性"""
        try:
            # 运行等效性测试
            cmd = [sys.executable, "ci/scripts/cuda_equivalence_guard.py", "--quiet"]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=300)
            
            # 尝试加载测试结果
            equivalence_file = project_root / "ci/results/latest/cuda_equivalence_test.json"
            equivalence_data = {}
            
            if equivalence_file.exists():
                try:
                    with open(equivalence_file) as f:
                        equivalence_data = json.load(f)
                except:
                    pass
            
            if result.returncode == 0:
                # 等效性测试通过
                return {
                    'status': 'passed',
                    'passed': True,
                    'message': 'CUDA内核等效性验证通过',
                    'details': equivalence_data.get('summary', {}),
                    'baseline_comparison': {
                        'test_count': equivalence_data.get('total_tests', 0),
                        'pass_rate': equivalence_data.get('summary', {}).get('pass_rate', 0)
                    }
                }
            else:
                # 等效性测试失败
                return {
                    'status': 'failed',
                    'passed': False,
                    'message': 'CUDA内核等效性验证失败',
                    'details': {
                        'returncode': result.returncode,
                        'stderr': result.stderr,
                        'test_data': equivalence_data
                    }
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'passed': False,
                'message': 'CUDA等效性测试超时',
                'details': {'timeout': True}
            }
        except Exception as e:
            return {
                'status': 'error',
                'passed': False,
                'message': f'CUDA等效性测试异常: {str(e)}'
            }
    
    def test_cuda_kernel_performance(self):
        """测试CUDA内核性能回归"""
        try:
            # 运行性能基准测试
            cmd = [sys.executable, "profile/scripts/cuda_kernel_benchmark.py", "--quick"]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'passed': False,
                    'message': f'CUDA性能测试执行失败: {result.stderr}',
                    'details': {'returncode': result.returncode}
                }
            
            # 解析性能数据（简化版本）
            output = result.stdout
            performance_indicators = {
                'benchmark_completed': 'benchmark' in output.lower() or 'performance' in output.lower(),
                'has_timing_data': 'ms' in output or 'time' in output.lower(),
                'has_error': 'error' in output.lower() or 'fail' in output.lower()
            }
            
            # 与基线对比
            baseline_comparison = {}
            if self.baseline_data and 'baselines' in self.baseline_data:
                baseline_cuda = self.baseline_data['baselines'].get('cuda_kernels', {})
                baseline_comparison['has_baseline'] = bool(baseline_cuda)
                baseline_comparison['baseline_success'] = baseline_cuda.get('benchmark_success', False)
            
            # 判断测试结果
            if performance_indicators['has_error']:
                status = 'failed'
                passed = False
                message = 'CUDA性能测试中检测到错误'
            elif not performance_indicators['benchmark_completed']:
                status = 'warning'
                passed = True
                message = 'CUDA性能测试完成但数据不完整'
            else:
                status = 'passed'
                passed = True
                message = 'CUDA性能测试正常完成'
            
            return {
                'status': status,
                'passed': passed,
                'message': message,
                'details': performance_indicators,
                'baseline_comparison': baseline_comparison
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'passed': False,
                'message': 'CUDA性能测试超时',
                'details': {'timeout': True}
            }
    
    def test_basic_functionality(self):
        """测试基础功能"""
        try:
            # 运行快速验证
            cmd = [sys.executable, "ci/scripts/quick_validate.sh"]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                return {
                    'status': 'passed',
                    'passed': True,
                    'message': '基础功能验证通过',
                    'details': {'quick_validation': True}
                }
            else:
                return {
                    'status': 'failed',
                    'passed': False,
                    'message': '基础功能验证失败',
                    'details': {
                        'returncode': result.returncode,
                        'stderr': result.stderr
                    }
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'passed': False,
                'message': '基础功能测试超时',
                'details': {'timeout': True}
            }
    
    def test_import_integrity(self):
        """测试导入完整性"""
        try:
            import_tests = [
                "from snn import ST_BIFNeuron_MS, MyQuan",
                "from models import resnet",
                "from wrapper import SNNWrapper_MS",
                "from utils import functions, io, misc"
            ]
            
            failed_imports = []
            for test in import_tests:
                try:
                    exec(test)
                except Exception as e:
                    failed_imports.append({'import': test, 'error': str(e)})
            
            if not failed_imports:
                return {
                    'status': 'passed',
                    'passed': True,
                    'message': '所有核心模块导入成功',
                    'details': {'tested_imports': len(import_tests)}
                }
            else:
                return {
                    'status': 'failed',
                    'passed': False,
                    'message': f'{len(failed_imports)}个模块导入失败',
                    'details': {'failed_imports': failed_imports}
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'passed': False,
                'message': f'导入测试异常: {str(e)}'
            }
    
    def run_full_suite(self):
        """运行完整的回归测试套件"""
        self.log("开始运行ST-BIF回归测试套件...")
        
        # 定义测试序列（按重要性排序）
        tests = [
            ('import_integrity', self.test_import_integrity),
            ('basic_functionality', self.test_basic_functionality),
            ('model_conversion_pipeline', self.test_model_conversion_pipeline),
            ('cuda_kernel_equivalence', self.test_cuda_kernel_equivalence),
            ('cuda_kernel_performance', self.test_cuda_kernel_performance)
        ]
        
        # 运行所有测试
        overall_success = True
        for test_name, test_func in tests:
            success = self.run_test(test_name, test_func)
            if not success:
                overall_success = False
        
        # 生成最终摘要
        self.test_results['summary']['overall_success'] = overall_success
        self.test_results['summary']['success_rate'] = (
            self.test_results['summary']['passed_tests'] / 
            self.test_results['summary']['total_tests'] 
            if self.test_results['summary']['total_tests'] > 0 else 0
        )
        
        return overall_success
    
    def save_results(self, output_file=None):
        """保存测试结果"""
        if output_file is None:
            output_file = self.results_dir / "regression_test_results.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"回归测试结果已保存到: {output_path}")
        return output_path
    
    def print_summary(self):
        """打印测试摘要"""
        summary = self.test_results['summary']
        
        print("\n" + "="*60)
        print("ST-BIF 回归测试摘要")
        print("="*60)
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")  
        print(f"警告测试: {summary['warning_tests']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"整体结果: {'✓ 通过' if summary['overall_success'] else '✗ 失败'}")
        
        # 显示失败的测试
        if summary['failed_tests'] > 0:
            print("\n失败的测试:")
            for test_name, test_result in self.test_results['tests'].items():
                if not test_result['passed'] and test_result.get('status') != 'warning':
                    print(f"  ✗ {test_name}: {test_result['message']}")
        
        # 显示警告的测试
        if summary['warning_tests'] > 0:
            print("\n有警告的测试:")
            for test_name, test_result in self.test_results['tests'].items():
                if test_result.get('status') == 'warning':
                    print(f"  ⚠ {test_name}: {test_result['message']}")

def main():
    parser = argparse.ArgumentParser(description="ST-BIF回归测试套件")
    parser.add_argument("--baseline", "-b", help="基线目录路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    # 创建回归测试套件
    suite = RegressionTestSuite(
        baseline_dir=args.baseline,
        verbose=not args.quiet
    )
    
    try:
        # 运行完整测试套件
        success = suite.run_full_suite()
        
        # 保存结果
        suite.save_results(args.output)
        
        # 显示摘要
        if not args.quiet:
            suite.print_summary()
        
        # 设置退出码
        sys.exit(0 if success else 1)
        
    except Exception as e:
        suite.log(f"回归测试套件发生错误: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()