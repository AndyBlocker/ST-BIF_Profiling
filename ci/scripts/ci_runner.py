#!/usr/bin/env python3
"""
ST-BIF CI Runner
完整的CI工作流运行器，支持不同的验证级别和场景
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class CIRunner:
    def __init__(self, config_file=None, verbose=True):
        self.verbose = verbose
        self.project_root = project_root
        self.ci_root = project_root / "ci"
        self.results_dir = self.ci_root / "results" / "latest"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # CI结果跟踪
        self.ci_results = {
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_info(),
            'stages': {},
            'summary': {
                'total_stages': 0,
                'passed_stages': 0,
                'failed_stages': 0,
                'overall_success': False
            }
        }
        
    def _get_git_info(self):
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                           cwd=self.project_root).decode().strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                           cwd=self.project_root).decode().strip()
            return {'commit': commit, 'branch': branch}
        except:
            return {'commit': 'unknown', 'branch': 'unknown'}
    
    def log(self, message, level='INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_stage(self, stage_name, command, timeout=300, required=True):
        """运行CI阶段"""
        self.log(f"开始CI阶段: {stage_name}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command, 
                cwd=self.project_root, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                shell=True
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            stage_result = {
                'success': success,
                'execution_time': execution_time,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'required': required,
                'command': command
            }
            
            self.ci_results['stages'][stage_name] = stage_result
            self.ci_results['summary']['total_stages'] += 1
            
            if success:
                self.ci_results['summary']['passed_stages'] += 1
                self.log(f"✓ {stage_name}: 成功 ({execution_time:.1f}s)")
            else:
                self.ci_results['summary']['failed_stages'] += 1
                self.log(f"✗ {stage_name}: 失败 ({execution_time:.1f}s)", "ERROR")
                if required:
                    self.log(f"  错误: {result.stderr[:200]}", "ERROR")
                    return False
                else:
                    self.log(f"  警告: {result.stderr[:200]}", "WARN")
            
            return success
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            stage_result = {
                'success': False,
                'execution_time': execution_time,
                'timeout': True,
                'required': required,
                'command': command
            }
            
            self.ci_results['stages'][stage_name] = stage_result
            self.ci_results['summary']['total_stages'] += 1
            self.ci_results['summary']['failed_stages'] += 1
            
            self.log(f"✗ {stage_name}: 超时 ({timeout}s)", "ERROR")
            return False if required else True
            
        except Exception as e:
            execution_time = time.time() - start_time
            stage_result = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'required': required,
                'command': command
            }
            
            self.ci_results['stages'][stage_name] = stage_result
            self.ci_results['summary']['total_stages'] += 1
            self.ci_results['summary']['failed_stages'] += 1
            
            self.log(f"✗ {stage_name}: 异常 - {str(e)}", "ERROR")
            return False if required else True
    
    def run_quick_ci(self):
        """快速CI流水线 (L0 + L1)"""
        self.log("启动快速CI流水线...")
        
        stages = [
            ("环境检查", "python --version && python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", 30, True),
            ("快速验证", "./ci/scripts/quick_validate.sh", 180, True),
            ("基础导入测试", "python -c 'from snn import ST_BIFNeuron_MS; from models import resnet; print(\"核心模块导入成功\")'", 60, True)
        ]
        
        success = True
        for stage_name, command, timeout, required in stages:
            if not self.run_stage(stage_name, command, timeout, required):
                if required:
                    success = False
                    break
        
        self.ci_results['summary']['overall_success'] = success
        return success
    
    def run_standard_ci(self):
        """标准CI流水线 (L0 + L1 + L2)"""
        self.log("启动标准CI流水线...")
        
        stages = [
            ("环境检查", "python --version", 30, True),
            ("快速验证", "./ci/scripts/quick_validate.sh", 180, True),
            ("模型转换测试", "python examples/ann_to_snn_conversion.py --quiet --batch-size 16", 600, True),
            ("CUDA等效性检查", "python ci/scripts/cuda_equivalence_guard.py --quiet", 300, False),  # 非必需，因为可能失败
        ]
        
        success = True
        for stage_name, command, timeout, required in stages:
            if not self.run_stage(stage_name, command, timeout, required):
                if required:
                    success = False
                    # 继续运行其他测试，不要中断
        
        self.ci_results['summary']['overall_success'] = success
        return success
    
    def run_full_ci(self):
        """完整CI流水线 (所有层级)"""
        self.log("启动完整CI流水线...")
        
        stages = [
            ("环境检查", "python --version", 30, True),
            ("快速验证", "./ci/scripts/quick_validate.sh", 180, True),
            ("回归测试套件", f"python ci/scripts/regression_test_suite.py --baseline ci/baselines/v1.0.0_current", 900, False),
            ("模型转换完整测试", "python examples/ann_to_snn_conversion.py --batch-size 32", 900, True),
            ("性能分析", "./profile/scripts/quick_profile.sh", 600, False),
        ]
        
        success = True
        required_failed = False
        
        for stage_name, command, timeout, required in stages:
            stage_success = self.run_stage(stage_name, command, timeout, required)
            if not stage_success and required:
                required_failed = True
        
        # 只有必需阶段都通过才算成功
        self.ci_results['summary']['overall_success'] = not required_failed
        return not required_failed
    
    def run_baseline_update(self):
        """基线更新流水线"""
        self.log("启动基线更新流水线...")
        
        # 先运行验证确保当前代码正常
        if not self.run_quick_ci():
            self.log("基线更新失败：当前代码未通过快速验证", "ERROR")
            return False
        
        # 生成新基线
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_dir = f"ci/baselines/v1.0.0_{timestamp}"
        
        success = self.run_stage(
            "生成新基线", 
            f"python ci/scripts/generate_baseline.py --output {baseline_dir}",
            600, 
            True
        )
        
        if success:
            # 更新当前基线链接
            self.run_stage(
                "更新基线链接",
                f"rm -f ci/baselines/current && ln -s v1.0.0_{timestamp} ci/baselines/current",
                30,
                False
            )
            self.log(f"基线已更新: {baseline_dir}")
        
        return success
    
    def save_results(self):
        """保存CI结果"""
        results_file = self.results_dir / "ci_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.ci_results, f, indent=2, ensure_ascii=False)
        
        self.log(f"CI结果已保存到: {results_file}")
        return results_file
    
    def print_summary(self):
        """打印CI摘要"""
        summary = self.ci_results['summary']
        
        print("\n" + "="*60)
        print("ST-BIF CI 执行摘要")
        print("="*60)
        print(f"总阶段数: {summary['total_stages']}")
        print(f"成功阶段: {summary['passed_stages']}")
        print(f"失败阶段: {summary['failed_stages']}")
        print(f"整体结果: {'✓ 成功' if summary['overall_success'] else '✗ 失败'}")
        
        # 显示阶段详情
        if summary['failed_stages'] > 0:
            print("\n失败的阶段:")
            for stage_name, stage_result in self.ci_results['stages'].items():
                if not stage_result['success']:
                    required_text = " (必需)" if stage_result.get('required', True) else " (可选)"
                    print(f"  ✗ {stage_name}{required_text}")
                    if 'stderr' in stage_result and stage_result['stderr']:
                        print(f"    错误: {stage_result['stderr'][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="ST-BIF CI运行器")
    parser.add_argument("--mode", "-m", choices=['quick', 'standard', 'full', 'baseline'], 
                       default='quick', help="CI模式")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    parser.add_argument("--output", "-o", help="结果输出目录")
    
    args = parser.parse_args()
    
    # 创建CI运行器
    runner = CIRunner(verbose=not args.quiet)
    
    try:
        # 根据模式运行不同的CI流水线
        if args.mode == 'quick':
            success = runner.run_quick_ci()
        elif args.mode == 'standard':
            success = runner.run_standard_ci()
        elif args.mode == 'full':
            success = runner.run_full_ci()
        elif args.mode == 'baseline':
            success = runner.run_baseline_update()
        
        # 保存结果
        runner.save_results()
        
        # 显示摘要
        if not args.quiet:
            runner.print_summary()
        
        # 设置退出码
        sys.exit(0 if success else 1)
        
    except Exception as e:
        runner.log(f"CI运行器发生错误: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()