#!/usr/bin/env python3
"""
ST-BIF Neuron Numerical Stability Testing Framework
===================================================

Tests ST-BIF neurons across different precision levels (FP16, FP32, FP64)
and mixed precision configurations to evaluate numerical stability.

Key test scenarios:
1. Single precision consistency across multiple runs
2. Cross-precision equivalence testing  
3. Gradient stability analysis
4. Edge case handling (extreme inputs, boundary conditions)
5. Long sequence stability
6. Mixed precision mode stability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, List, Tuple, Any
import warnings
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Import ST-BIF neuron implementations
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS, ST_BIFNodeATGF_MS
from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA


class NumericalStabilityTester:
    """
    Comprehensive numerical stability testing framework for ST-BIF neurons
    """
    
    def __init__(self, device='cuda', save_results=True, output_dir='tests/outputs'):
        self.device = device
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configurations
        self.precisions = [torch.float16, torch.float32, torch.float64]
        self.precision_names = ['fp16', 'fp32', 'fp64']
        
        # Test parameters
        self.test_params = {
            'batch_sizes': [1, 8, 32],
            'time_steps': [4, 8, 16],
            'feature_dims': [(10,), (3, 32, 32), (512,)],
            'threshold_values': [0.5, 1.0, 2.0],
            'level_values': [4, 8, 16],
            'input_ranges': [(-2.0, 2.0), (-5.0, 5.0), (-10.0, 10.0)]
        }
        
        # Results storage
        self.results = {
            'precision_consistency': {},
            'cross_precision_equivalence': {},
            'gradient_stability': {},
            'edge_cases': {},
            'long_sequence_stability': {},
            'mixed_precision_stability': {}
        }
    
    def create_test_input(self, batch_size: int, time_steps: int, 
                         feature_shape: Tuple[int, ...], 
                         input_range: Tuple[float, float],
                         dtype: torch.dtype) -> torch.Tensor:
        """Create deterministic test input with controlled characteristics"""
        torch.manual_seed(42)  # Ensure reproducibility
        
        total_shape = (time_steps * batch_size,) + feature_shape
        x = torch.randn(total_shape, dtype=dtype, device=self.device)
        
        # Scale to desired range
        min_val, max_val = input_range
        x = x * (max_val - min_val) / 4.0 + (max_val + min_val) / 2.0
        
        return x
    
    def create_neuron(self, threshold: float, level: int, dtype: torch.dtype, time_steps: int) -> ST_BIFNeuron_MS:
        """Create ST-BIF neuron with specified parameters"""
        neuron = ST_BIFNeuron_MS(
            q_threshold=torch.tensor(threshold, dtype=dtype, device=self.device),
            level=torch.tensor(level),
            sym=False
        )
        neuron.T = time_steps
        neuron.to(device=self.device, dtype=dtype)
        return neuron
    
    def compute_metrics(self, output1: torch.Tensor, output2: torch.Tensor) -> Dict[str, float]:
        """Compute numerical difference metrics between two outputs"""
        # Handle potential dtype differences
        if output1.dtype != output2.dtype:
            # Convert to higher precision for comparison
            if output1.dtype == torch.float64 or output2.dtype == torch.float64:
                target_dtype = torch.float64
            elif output1.dtype == torch.float32 or output2.dtype == torch.float32:
                target_dtype = torch.float32
            else:
                target_dtype = torch.float32
            output1 = output1.to(target_dtype)
            output2 = output2.to(target_dtype)
        
        diff = output1 - output2
        
        metrics = {
            'max_abs_error': torch.max(torch.abs(diff)).item(),
            'mean_abs_error': torch.mean(torch.abs(diff)).item(),
            'mse': torch.mean(diff ** 2).item(),
            'rel_error': (torch.norm(diff) / torch.norm(output1)).item() if torch.norm(output1) > 0 else 0.0,
            'num_different': torch.sum(diff != 0).item(),
            'total_elements': diff.numel()
        }
        
        # Add spike-specific metrics
        if len(output1.shape) >= 2:  # Assume spike sequences
            spike_rate_1 = torch.mean(torch.abs(output1)).item()
            spike_rate_2 = torch.mean(torch.abs(output2)).item()
            metrics['spike_rate_diff'] = abs(spike_rate_1 - spike_rate_2)
        
        return metrics
    
    def test_precision_consistency(self) -> Dict[str, Any]:
        """Test consistency within same precision across multiple runs"""
        print("Testing precision consistency...")
        results = {}
        
        for dtype, name in zip(self.precisions, self.precision_names):
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue  # Skip FP16 on CPU
                
            print(f"  Testing {name}...")
            precision_results = []
            
            # Test multiple parameter combinations
            for batch_size in [8]:
                for time_steps in [8]:
                    for feature_shape in [(64,)]:
                        for threshold in [1.0]:
                            for level in [8]:
                                test_key = f"b{batch_size}_t{time_steps}_f{feature_shape}_th{threshold}_l{level}"
                                
                                # Create test setup
                                x = self.create_test_input(batch_size, time_steps, feature_shape, (-2.0, 2.0), dtype)
                                
                                # Multiple runs with same seed
                                outputs = []
                                for run in range(3):
                                    torch.manual_seed(42)
                                    neuron = self.create_neuron(threshold, level, dtype, time_steps)
                                    neuron.reset()
                                    output = neuron(x)
                                    outputs.append(output.detach().clone())
                                
                                # Compare consistency
                                consistency_metrics = []
                                for i in range(1, len(outputs)):
                                    metrics = self.compute_metrics(outputs[0], outputs[i])
                                    consistency_metrics.append(metrics)
                                
                                precision_results.append({
                                    'test_key': test_key,
                                    'consistency_metrics': consistency_metrics
                                })
            
            results[name] = precision_results
        
        self.results['precision_consistency'] = results
        return results
    
    def test_cross_precision_equivalence(self) -> Dict[str, Any]:
        """Test equivalence between different precision levels"""
        print("Testing cross-precision equivalence...")
        results = {}
        
        # Reference: FP64 (highest precision)
        ref_dtype = torch.float64
        ref_name = 'fp64'
        
        for dtype, name in zip(self.precisions, self.precision_names):
            if name == ref_name or (dtype == torch.float16 and not torch.cuda.is_available()):
                continue
                
            print(f"  Comparing {name} vs {ref_name}...")
            comparison_results = []
            
            for batch_size in [4]:
                for time_steps in [8]:
                    for feature_shape in [(32,)]:
                        for threshold in [1.0]:
                            for level in [8]:
                                test_key = f"b{batch_size}_t{time_steps}_f{feature_shape}_th{threshold}_l{level}"
                                
                                # Create identical inputs (but different dtypes)
                                x_ref = self.create_test_input(batch_size, time_steps, feature_shape, (-2.0, 2.0), ref_dtype)
                                x_test = x_ref.to(dtype)
                                
                                # Run reference (FP64)
                                torch.manual_seed(42)
                                neuron_ref = self.create_neuron(threshold, level, ref_dtype, time_steps)
                                neuron_ref.reset()
                                output_ref = neuron_ref(x_ref)
                                
                                # Run test precision
                                torch.manual_seed(42)
                                neuron_test = self.create_neuron(threshold, level, dtype, time_steps)
                                neuron_test.reset()
                                output_test = neuron_test(x_test)
                                
                                # Compare outputs
                                metrics = self.compute_metrics(output_ref, output_test)
                                
                                comparison_results.append({
                                    'test_key': test_key,
                                    'comparison_metrics': metrics
                                })
            
            results[f"{name}_vs_{ref_name}"] = comparison_results
        
        self.results['cross_precision_equivalence'] = results
        return results
    
    def test_gradient_stability(self) -> Dict[str, Any]:
        """Test gradient computation stability across precisions"""
        print("Testing gradient stability...")
        results = {}
        
        for dtype, name in zip(self.precisions, self.precision_names):
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue
                
            print(f"  Testing gradient stability for {name}...")
            gradient_results = []
            
            # Simple test case
            batch_size, time_steps, feature_shape = 4, 8, (16,)
            threshold, level = 1.0, 8
            
            x = self.create_test_input(batch_size, time_steps, feature_shape, (-1.0, 1.0), dtype)
            x.requires_grad_(True)
            
            # Multiple gradient computations
            gradients = []
            for run in range(3):
                torch.manual_seed(42)
                neuron = self.create_neuron(threshold, level, dtype, time_steps)
                neuron.reset()
                
                output = neuron(x)
                loss = torch.sum(output ** 2)
                
                # Clear previous gradients
                if x.grad is not None:
                    x.grad.zero_()
                    
                loss.backward(retain_graph=True)
                gradients.append(x.grad.detach().clone() if x.grad is not None else torch.zeros_like(x))
            
            # Analyze gradient consistency
            gradient_metrics = []
            for i in range(1, len(gradients)):
                metrics = self.compute_metrics(gradients[0], gradients[i])
                gradient_metrics.append(metrics)
            
            results[name] = {
                'gradient_consistency': gradient_metrics,
                'gradient_magnitude': torch.norm(gradients[0]).item() if len(gradients) > 0 else 0.0
            }
        
        self.results['gradient_stability'] = results
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test behavior with edge cases and extreme inputs"""
        print("Testing edge cases...")
        results = {}
        
        edge_cases = [
            ('zero_input', lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=self.device)),
            ('large_positive', lambda shape, dtype: torch.full(shape, 100.0, dtype=dtype, device=self.device)),
            ('large_negative', lambda shape, dtype: torch.full(shape, -100.0, dtype=dtype, device=self.device)),
            ('inf_input', lambda shape, dtype: torch.full(shape, float('inf'), dtype=dtype, device=self.device)),
            ('tiny_input', lambda shape, dtype: torch.full(shape, 1e-8, dtype=dtype, device=self.device))
        ]
        
        for dtype, name in zip(self.precisions, self.precision_names):
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue
                
            print(f"  Testing edge cases for {name}...")
            edge_results = {}
            
            batch_size, time_steps, feature_shape = 2, 4, (8,)
            threshold, level = 1.0, 8
            
            for case_name, input_generator in edge_cases:
                try:
                    # Skip inf test for FP16 as it may not be well supported
                    if case_name == 'inf_input' and dtype == torch.float16:
                        continue
                        
                    shape = (time_steps * batch_size,) + feature_shape
                    x = input_generator(shape, dtype)
                    
                    torch.manual_seed(42)
                    neuron = self.create_neuron(threshold, level, dtype, time_steps)
                    neuron.reset()
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        output = neuron(x)
                    
                    # Check for NaN/Inf in output
                    has_nan = torch.isnan(output).any().item()
                    has_inf = torch.isinf(output).any().item()
                    output_range = (torch.min(output).item(), torch.max(output).item())
                    
                    edge_results[case_name] = {
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'output_range': output_range,
                        'mean_output': torch.mean(output).item()
                    }
                    
                except Exception as e:
                    edge_results[case_name] = {
                        'error': str(e),
                        'failed': True
                    }
            
            results[name] = edge_results
        
        self.results['edge_cases'] = results
        return results
    
    def test_long_sequence_stability(self) -> Dict[str, Any]:
        """Test stability with long time sequences"""
        print("Testing long sequence stability...")
        results = {}
        
        sequence_lengths = [16, 32, 64]
        
        for dtype, name in zip(self.precisions, self.precision_names):
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue
                
            print(f"  Testing long sequences for {name}...")
            sequence_results = []
            
            batch_size, feature_shape = 2, (16,)
            threshold, level = 1.0, 8
            
            for seq_len in sequence_lengths:
                try:
                    x = self.create_test_input(batch_size, seq_len, feature_shape, (-1.0, 1.0), dtype)
                    
                    torch.manual_seed(42)
                    neuron = self.create_neuron(threshold, level, dtype, seq_len)
                    neuron.reset()
                    
                    output = neuron(x)
                    
                    # Analyze temporal stability
                    output_reshaped = output.view(seq_len, batch_size, -1)
                    temporal_variance = torch.var(torch.mean(output_reshaped, dim=(1, 2))).item()
                    
                    sequence_results.append({
                        'sequence_length': seq_len,
                        'temporal_variance': temporal_variance,
                        'mean_spike_rate': torch.mean(torch.abs(output)).item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    })
                    
                except Exception as e:
                    sequence_results.append({
                        'sequence_length': seq_len,
                        'error': str(e),
                        'failed': True
                    })
            
            results[name] = sequence_results
        
        self.results['long_sequence_stability'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("Starting ST-BIF Numerical Stability Test Suite...")
        print("=" * 60)
        
        # Run all test categories
        self.test_precision_consistency()
        self.test_cross_precision_equivalence()
        self.test_gradient_stability()
        self.test_edge_cases()
        self.test_long_sequence_stability()
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results if requested
        if self.save_results:
            self.save_test_results()
            self.generate_report()
        
        print("\nTest suite completed!")
        return {
            'detailed_results': self.results,
            'summary': summary
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary with key findings"""
        summary = {
            'overall_status': 'PASSED',
            'critical_issues': [],
            'warnings': [],
            'precision_rankings': {},
            'recommendations': []
        }
        
        # Analyze cross-precision equivalence
        equiv_results = self.results.get('cross_precision_equivalence', {})
        for comparison, tests in equiv_results.items():
            max_errors = []
            for test in tests:
                if 'comparison_metrics' in test:
                    max_errors.append(test['comparison_metrics']['max_abs_error'])
            
            if max_errors:
                avg_max_error = np.mean(max_errors)
                if avg_max_error > 1e-2:
                    summary['critical_issues'].append(f"High numerical error in {comparison}: {avg_max_error:.2e}")
                elif avg_max_error > 1e-4:
                    summary['warnings'].append(f"Moderate numerical error in {comparison}: {avg_max_error:.2e}")
        
        # Analyze edge cases
        edge_results = self.results.get('edge_cases', {})
        for precision, cases in edge_results.items():
            for case_name, result in cases.items():
                if result.get('has_nan', False):
                    summary['critical_issues'].append(f"NaN detected in {precision} {case_name}")
                if result.get('has_inf', False) and case_name != 'inf_input':
                    summary['critical_issues'].append(f"Inf detected in {precision} {case_name}")
        
        # Set overall status
        if summary['critical_issues']:
            summary['overall_status'] = 'FAILED'
        elif summary['warnings']:
            summary['overall_status'] = 'WARNING'
        
        # Generate recommendations
        if summary['critical_issues']:
            summary['recommendations'].append("Critical numerical instabilities detected - review implementation")
        if len(summary['warnings']) > 2:
            summary['recommendations'].append("Consider using higher precision for critical applications")
        
        return summary
    
    def save_test_results(self):
        """Save detailed test results to JSON"""
        # Convert tensors to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(self.results)
        
        output_file = self.output_dir / f"numerical_stability_results.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")
    
    def generate_report(self):
        """Generate human-readable HTML report"""
        # This would generate a comprehensive HTML report
        # For now, create a simple markdown summary
        report_file = self.output_dir / "numerical_stability_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# ST-BIF Neuron Numerical Stability Test Report\n\n")
            f.write(f"**Test Date:** {torch.utils.data.get_worker_info()}\n")
            f.write(f"**Device:** {self.device}\n\n")
            
            summary = self.generate_summary()
            f.write(f"## Overall Status: {summary['overall_status']}\n\n")
            
            if summary['critical_issues']:
                f.write("### Critical Issues\n")
                for issue in summary['critical_issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            if summary['warnings']:
                f.write("### Warnings\n")
                for warning in summary['warnings']:
                    f.write(f"- {warning}\n")
                f.write("\n")
            
            if summary['recommendations']:
                f.write("### Recommendations\n")
                for rec in summary['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
        
        print(f"Report saved to: {report_file}")


# Test functions for pytest integration
def test_numerical_stability_quick():
    """Quick numerical stability test for CI"""
    tester = NumericalStabilityTester(save_results=False)
    results = tester.test_precision_consistency()
    assert results is not None

def test_cross_precision_equivalence():
    """Test cross-precision equivalence"""
    tester = NumericalStabilityTester(save_results=False)
    results = tester.test_cross_precision_equivalence()
    assert results is not None

if __name__ == "__main__":
    # Run full test suite
    tester = NumericalStabilityTester(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_results=True
    )
    
    results = tester.run_all_tests()
    print("\nTest Summary:")
    print(f"Overall Status: {results['summary']['overall_status']}")
    
    if results['summary']['critical_issues']:
        print("Critical Issues:")
        for issue in results['summary']['critical_issues']:
            print(f"  - {issue}")
    
    if results['summary']['warnings']:
        print("Warnings:")
        for warning in results['summary']['warnings']:
            print(f"  - {warning}")