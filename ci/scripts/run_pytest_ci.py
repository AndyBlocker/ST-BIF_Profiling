#!/usr/bin/env python3
"""
Pytest CI Runner - Integration of pytest test suite with ST-BIF CI framework
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced report generator
try:
    from ci.scripts.generate_report import CIReportGenerator
except ImportError:
    CIReportGenerator = None


class PytestCIRunner:
    """Run pytest test suite with CI integration"""
    
    def __init__(self, output_dir=None, verbose=True):
        self.verbose = verbose
        self.project_root = project_root
        self.output_dir = Path(output_dir) if output_dir else project_root / "ci" / "results" / "latest"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_runs': {},
            'summary': {
                'total_test_suites': 0,
                'passed_test_suites': 0,
                'failed_test_suites': 0,
                'overall_success': False
            }
        }
    
    def log(self, message, level='INFO'):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_pytest_command(self, test_path="", markers="", extra_args=None):
        """Run pytest with specified parameters"""
        cmd = ["python", "-m", "pytest"]
        
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append("tests/")
        
        if markers:
            cmd.extend(["-m", markers])
        
        # Standard pytest args
        cmd.extend([
            "-v",
            "--tb=short", 
            "--color=yes",
            "--durations=10",
            f"--junitxml={self.output_dir}/pytest-results.xml"
        ])
        
        if extra_args:
            cmd.extend(extra_args)
        
        return cmd
    
    def execute_test_suite(self, name, cmd, timeout=300):
        """Execute a pytest test suite"""
        self.log(f"Running {name} test suite...")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # Parse pytest output for more details
            test_summary = self._parse_pytest_output(result.stdout)
            
            suite_result = {
                'success': success,
                'execution_time': execution_time,
                'returncode': result.returncode,
                'command': ' '.join(cmd),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_summary': test_summary
            }
            
            self.results['test_runs'][name] = suite_result
            self.results['summary']['total_test_suites'] += 1
            
            if success:
                self.results['summary']['passed_test_suites'] += 1
                self.log(f"✓ {name}: PASSED ({execution_time:.1f}s, {test_summary.get('passed', 0)} tests)")
            else:
                self.results['summary']['failed_test_suites'] += 1
                self.log(f"✗ {name}: FAILED ({execution_time:.1f}s)", "ERROR")
                if test_summary.get('failed', 0) > 0:
                    self.log(f"  Failed tests: {test_summary['failed']}", "ERROR")
                if test_summary.get('errors', 0) > 0:
                    self.log(f"  Errors: {test_summary['errors']}", "ERROR")
            
            return success
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.results['test_runs'][name] = {
                'success': False,
                'execution_time': execution_time,
                'timeout': True,
                'command': ' '.join(cmd)
            }
            self.results['summary']['total_test_suites'] += 1
            self.results['summary']['failed_test_suites'] += 1
            self.log(f"✗ {name}: TIMEOUT after {timeout}s", "ERROR")
            return False
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results['test_runs'][name] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'command': ' '.join(cmd)
            }
            self.results['summary']['total_test_suites'] += 1
            self.results['summary']['failed_test_suites'] += 1
            self.log(f"✗ {name}: ERROR - {str(e)}", "ERROR")
            return False
    
    def _parse_pytest_output(self, output):
        """Parse pytest output to extract test counts"""
        summary = {'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}
        
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Look for summary line like "5 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            summary['passed'] = int(parts[i-1])
                        except:
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            summary['failed'] = int(parts[i-1])
                        except:
                            pass
                    elif part == 'error' and i > 0:
                        try:
                            summary['errors'] = int(parts[i-1])
                        except:
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            summary['skipped'] = int(parts[i-1])
                        except:
                            pass
                break
        
        return summary
    
    def run_quick_tests(self):
        """Run quick test suite (imports and basic functionality)"""
        self.log("Running quick pytest suite...")
        
        test_suites = [
            ("Import Tests", self.run_pytest_command("tests/test_imports.py")),
            ("Basic CUDA Tests", self.run_pytest_command("tests/test_cuda_kernels.py", "not slow and not performance")),
        ]
        
        all_passed = True
        for name, cmd in test_suites:
            success = self.execute_test_suite(name, cmd, timeout=180)
            if not success:
                all_passed = False
        
        self.results['summary']['overall_success'] = all_passed
        return all_passed
    
    def run_standard_tests(self):
        """Run standard test suite (excluding slow tests)"""
        self.log("Running standard pytest suite...")
        
        test_suites = [
            ("Import Tests", self.run_pytest_command("tests/test_imports.py")),
            ("CUDA Equivalence Tests", self.run_pytest_command("tests/test_cuda_kernels.py", "equivalence and not slow")),
            ("Model Conversion Tests", self.run_pytest_command("tests/test_model_conversion.py", "not slow")),
        ]
        
        all_passed = True
        for name, cmd in test_suites:
            success = self.execute_test_suite(name, cmd, timeout=300)
            if not success:
                all_passed = False
        
        self.results['summary']['overall_success'] = all_passed
        return all_passed
    
    def run_full_tests(self):
        """Run full test suite (including slow tests)"""
        self.log("Running full pytest suite...")
        
        test_suites = [
            ("Import Tests", self.run_pytest_command("tests/test_imports.py")),
            ("CUDA Kernel Tests", self.run_pytest_command("tests/test_cuda_kernels.py")),
            ("Model Conversion Tests", self.run_pytest_command("tests/test_model_conversion.py")),
            ("Performance Benchmarks", self.run_pytest_command("tests/benchmark_cuda_kernels.py", "performance")),
        ]
        
        all_passed = True
        for name, cmd in test_suites:
            # Longer timeout for full tests
            timeout = 600 if "Performance" in name else 300
            success = self.execute_test_suite(name, cmd, timeout=timeout)
            if not success:
                all_passed = False
        
        self.results['summary']['overall_success'] = all_passed
        return all_passed
    
    def run_specific_markers(self, markers):
        """Run tests with specific markers"""
        self.log(f"Running pytest with markers: {markers}")
        
        cmd = self.run_pytest_command(markers=markers)
        success = self.execute_test_suite(f"Tests ({markers})", cmd, timeout=600)
        
        self.results['summary']['overall_success'] = success
        return success
    
    def save_results(self):
        """Save test results to file"""
        results_file = self.output_dir / "pytest_ci_results.json"
        
        # Add git information to results
        try:
            import subprocess
            git_branch = subprocess.check_output(['git', 'branch', '--show-current'], 
                                               cwd=self.project_root, text=True).strip()
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               cwd=self.project_root, text=True).strip()
            self.results['git_commit'] = {
                'branch': git_branch,
                'commit': git_commit
            }
        except:
            self.results['git_commit'] = {
                'branch': 'unknown',
                'commit': 'unknown'
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"Pytest CI results saved to: {results_file}")
        return results_file
    
    def generate_enhanced_reports(self):
        """Generate enhanced reports using the new report generator"""
        if CIReportGenerator is None:
            self.log("Enhanced report generator not available, falling back to basic summary", "WARNING")
            return False
        
        try:
            generator = CIReportGenerator()
            
            # Generate console report
            console_report = generator.generate_console_report(self.results)
            print("\n" + console_report)
            
            # Save enhanced reports
            # HTML report
            html_report = generator.generate_html_report(self.results)
            html_path = self.output_dir / "enhanced_report.html"
            html_path.write_text(html_report, encoding='utf-8')
            self.log(f"Enhanced HTML report saved: {html_path}")
            
            # Markdown report  
            md_report = generator.generate_markdown_report(self.results)
            md_path = self.output_dir / "enhanced_report.md"
            md_path.write_text(md_report, encoding='utf-8')
            self.log(f"Enhanced Markdown report saved: {md_path}")
            
            return True
            
        except Exception as e:
            self.log(f"Error generating enhanced reports: {e}", "ERROR")
            return False
    
    def print_summary(self):
        """Print test summary - try enhanced report first, then fallback"""
        # Try enhanced reporting first
        if self.generate_enhanced_reports():
            return
        
        # Fallback to basic summary
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("Pytest CI Summary")
        print("="*60)
        print(f"Total test suites: {summary['total_test_suites']}")
        print(f"Passed test suites: {summary['passed_test_suites']}")
        print(f"Failed test suites: {summary['failed_test_suites']}")
        print(f"Overall result: {'✓ PASSED' if summary['overall_success'] else '✗ FAILED'}")
        
        # Show failed test details
        if summary['failed_test_suites'] > 0:
            print("\nFailed test suites:")
            for suite_name, suite_result in self.results['test_runs'].items():
                if not suite_result['success']:
                    print(f"  ✗ {suite_name}")
                    if 'test_summary' in suite_result:
                        ts = suite_result['test_summary']
                        if ts.get('failed', 0) > 0:
                            print(f"    Failed tests: {ts['failed']}")
                        if ts.get('errors', 0) > 0:
                            print(f"    Errors: {ts['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Pytest CI Runner for ST-BIF")
    parser.add_argument("--mode", "-m", choices=['quick', 'standard', 'full'], 
                       default='quick', help="Test mode")
    parser.add_argument("--markers", help="Run tests with specific pytest markers")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--report-format", choices=['enhanced', 'basic'], 
                       default='enhanced', help="Report format to use")
    
    args = parser.parse_args()
    
    # Create pytest CI runner
    runner = PytestCIRunner(
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    try:
        # Run tests based on mode
        if args.markers:
            success = runner.run_specific_markers(args.markers)
        elif args.mode == 'quick':
            success = runner.run_quick_tests()
        elif args.mode == 'standard':
            success = runner.run_standard_tests()
        elif args.mode == 'full':
            success = runner.run_full_tests()
        
        # Save results and show summary
        runner.save_results()
        
        if not args.quiet:
            runner.print_summary()
        
        # Set exit code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        runner.log(f"Pytest CI runner error: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()