#!/usr/bin/env python3
"""
Final ST-BIF Numerical Stability Test
=====================================

This test correctly evaluates ST-BIF neuron stability across precision levels,
accounting for legitimate numerical differences due to floating-point precision.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

class ST_BIFNumericalStabilityTest:
    """
    Comprehensive numerical stability test that properly handles precision differences
    """
    
    def __init__(self, device='cuda', output_dir='tests/outputs/final_stability'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tolerance bounds based on empirical analysis
        self.tolerances = {
            'spike_rate_tolerance': 0.1,      # 10% difference in spike rate
            'pattern_similarity_threshold': 0.8,  # 80% pattern similarity
            'gradient_relative_tolerance': 0.01,   # 1% relative gradient difference
            'boundary_case_tolerance': 0.5,        # Higher tolerance for boundary cases
        }
        
        self.results = {
            'summary': {},
            'detailed_tests': {},
            'recommendations': []
        }
    
    def test_precision_consistency_with_tolerance(self):
        """Test precision consistency with appropriate tolerance bounds"""
        print("=== Precision Consistency Test (With Tolerance) ===")
        
        test_cases = [
            {'name': 'normal_range', 'input_scale': 1.0, 'description': 'Normal input range'},
            {'name': 'small_signals', 'input_scale': 0.1, 'description': 'Small signal inputs'},
            {'name': 'large_signals', 'input_scale': 3.0, 'description': 'Large signal inputs'},
            {'name': 'boundary_case', 'input_scale': 0.99, 'description': 'Near-threshold inputs'},
        ]
        
        results = {}
        
        for case in test_cases:
            print(f"\nTesting {case['name']}: {case['description']}")
            
            # Generate test data
            batch_size, time_steps, feature_dim = 8, 8, 16
            torch.manual_seed(42)
            x_base = torch.randn(time_steps * batch_size, feature_dim) * case['input_scale']
            
            case_results = {}
            outputs = {}
            
            # Test each precision
            for dtype_name, dtype in [('fp16', torch.float16), ('fp32', torch.float32), ('fp64', torch.float64)]:
                if dtype == torch.float16 and self.device == 'cpu':
                    continue
                
                print(f"  {dtype_name}: ", end='')
                
                x = x_base.to(dtype=dtype, device=self.device)
                
                torch.manual_seed(42)
                neuron = ST_BIFNeuron_MS(
                    q_threshold=torch.tensor(1.0, dtype=dtype, device=self.device),
                    level=torch.tensor(8),
                    sym=False
                )
                neuron.T = time_steps
                neuron.to(device=self.device)
                neuron.reset()
                
                output = neuron(x)
                outputs[dtype_name] = output.cpu().float()
                
                # Compute metrics
                spike_rate = torch.mean(torch.abs(output)).item()
                num_spikes = torch.sum(output != 0).item()
                total_elements = output.numel()
                
                case_results[dtype_name] = {
                    'spike_rate': spike_rate,
                    'num_spikes': num_spikes,
                    'total_elements': total_elements,
                    'sparsity': 1.0 - (num_spikes / total_elements)
                }
                
                print(f"spike_rate={spike_rate:.4f}, sparsity={case_results[dtype_name]['sparsity']:.3f}")
            
            # Compare precisions
            precisions = list(outputs.keys())
            comparisons = {}
            
            for i in range(len(precisions)):
                for j in range(i + 1, len(precisions)):
                    p1, p2 = precisions[i], precisions[j]
                    
                    # Compute similarity metrics
                    out1, out2 = outputs[p1], outputs[p2]
                    
                    # Pattern similarity (element-wise agreement)
                    pattern_agreement = torch.mean(((out1 != 0) == (out2 != 0)).float()).item()
                    
                    # Spike rate similarity
                    rate1 = case_results[p1]['spike_rate']
                    rate2 = case_results[p2]['spike_rate']
                    rate_similarity = 1.0 - abs(rate1 - rate2) / max(rate1, rate2, 1e-6)
                    
                    # Absolute difference
                    abs_diff = torch.mean(torch.abs(out1 - out2)).item()
                    max_diff = torch.max(torch.abs(out1 - out2)).item()
                    
                    comparison_key = f"{p1}_vs_{p2}"
                    comparisons[comparison_key] = {
                        'pattern_agreement': pattern_agreement,
                        'rate_similarity': rate_similarity,
                        'mean_abs_diff': abs_diff,
                        'max_abs_diff': max_diff
                    }
                    
                    # Assess result
                    if pattern_agreement >= self.tolerances['pattern_similarity_threshold']:
                        status = "✅ PASS"
                    elif rate_similarity >= 1.0 - self.tolerances['spike_rate_tolerance']:
                        status = "⚠️ ACCEPTABLE"
                    else:
                        status = "❌ FAIL"
                    
                    print(f"  {comparison_key}: pattern={pattern_agreement:.3f}, rate_sim={rate_similarity:.3f} → {status}")
            
            case_results['comparisons'] = comparisons
            results[case['name']] = case_results
        
        self.results['detailed_tests']['precision_consistency'] = results
        return results
    
    def test_boundary_sensitivity(self):
        """Test behavior near critical boundaries"""
        print("\n=== Boundary Sensitivity Test ===")
        
        # Test values very close to spike threshold
        threshold = 1.0
        boundary_tests = [
            threshold - 1e-6,   # Just below threshold
            threshold - 1e-7,   # Very close below
            threshold,          # Exactly at threshold
            threshold + 1e-7,   # Very close above
            threshold + 1e-6,   # Just above threshold
        ]
        
        results = {}
        
        for test_val in boundary_tests:
            print(f"\nTesting boundary value: {test_val:.8f}")
            
            val_results = {}
            
            for dtype_name, dtype in [('fp32', torch.float32), ('fp64', torch.float64)]:
                x = torch.tensor([[test_val]], dtype=dtype, device=self.device)
                
                neuron = ST_BIFNeuron_MS(
                    q_threshold=torch.tensor(threshold, dtype=dtype, device=self.device),
                    level=torch.tensor(8),
                    sym=False
                )
                neuron.T = 1
                neuron.to(device=self.device)
                neuron.reset()
                
                output = neuron(x)
                spike = output.item()
                
                val_results[dtype_name] = {
                    'input_value': test_val,
                    'actual_input': x.item(),  # What precision was actually used
                    'spike_output': spike,
                    'threshold': threshold
                }
                
                print(f"  {dtype_name}: input={x.item():.10f} → spike={spike:.1f}")
            
            # Check consistency
            fp32_spike = val_results['fp32']['spike_output']
            fp64_spike = val_results['fp64']['spike_output']
            
            if fp32_spike == fp64_spike:
                consistency = "✅ CONSISTENT"
            else:
                consistency = "⚠️ PRECISION_DEPENDENT"
            
            print(f"  Consistency: {consistency}")
            
            val_results['consistency'] = consistency
            results[f"boundary_{test_val:.8f}"] = val_results
        
        self.results['detailed_tests']['boundary_sensitivity'] = results
        return results
    
    def test_gradient_stability_with_tolerance(self):
        """Test gradient stability with appropriate tolerance"""
        print("\n=== Gradient Stability Test ===")
        
        batch_size, time_steps, feature_dim = 4, 4, 8
        
        results = {}
        
        for trial in range(3):
            print(f"\nTrial {trial + 1}:")
            
            torch.manual_seed(42 + trial)
            x_base = torch.randn(time_steps * batch_size, feature_dim)
            
            trial_results = {}
            gradients = {}
            
            for dtype_name, dtype in [('fp32', torch.float32), ('fp64', torch.float64)]:
                x = x_base.to(dtype=dtype, device=self.device)
                x.requires_grad_(True)
                
                torch.manual_seed(42 + trial)
                neuron = ST_BIFNeuron_MS(
                    q_threshold=torch.tensor(1.0, dtype=dtype, device=self.device),
                    level=torch.tensor(8),
                    sym=False
                )
                neuron.T = time_steps
                neuron.to(device=self.device)
                neuron.reset()
                
                output = neuron(x)
                loss = torch.sum(output ** 2)
                
                # Clear any existing gradients
                if x.grad is not None:
                    x.grad.zero_()
                
                loss.backward()
                
                if x.grad is not None:
                    grad = x.grad.clone().cpu().float()
                    grad_norm = torch.norm(grad).item()
                    
                    gradients[dtype_name] = grad
                    trial_results[dtype_name] = {
                        'gradient_norm': grad_norm,
                        'loss': loss.item(),
                        'output_spike_rate': torch.mean(torch.abs(output)).item()
                    }
                    
                    print(f"  {dtype_name}: grad_norm={grad_norm:.6f}, loss={loss.item():.6f}")
                else:
                    print(f"  {dtype_name}: No gradients computed")
                    trial_results[dtype_name] = {'error': 'No gradients'}
            
            # Compare gradients if both exist
            if 'fp32' in gradients and 'fp64' in gradients:
                grad_fp32 = gradients['fp32']
                grad_fp64 = gradients['fp64']
                
                # Relative difference
                grad_diff = torch.abs(grad_fp32 - grad_fp64)
                max_grad_diff = torch.max(grad_diff).item()
                mean_grad_diff = torch.mean(grad_diff).item()
                
                # Relative error
                grad_norm_fp64 = torch.norm(grad_fp64).item()
                relative_error = torch.norm(grad_diff).item() / (grad_norm_fp64 + 1e-10)
                
                comparison = {
                    'max_abs_diff': max_grad_diff,
                    'mean_abs_diff': mean_grad_diff,
                    'relative_error': relative_error
                }
                
                # Assess gradient stability
                if relative_error < self.tolerances['gradient_relative_tolerance']:
                    grad_status = "✅ STABLE"
                elif relative_error < 0.1:  # 10%
                    grad_status = "⚠️ ACCEPTABLE"
                else:
                    grad_status = "❌ UNSTABLE"
                
                print(f"  Gradient comparison: rel_error={relative_error:.4f} → {grad_status}")
                
                trial_results['comparison'] = comparison
                trial_results['status'] = grad_status
            
            results[f"trial_{trial + 1}"] = trial_results
        
        self.results['detailed_tests']['gradient_stability'] = results
        return results
    
    def generate_final_assessment(self):
        """Generate final assessment and recommendations"""
        print("\n=== Final Assessment ===")
        
        # Analyze results
        issues = []
        warnings = []
        passes = []
        
        # Check precision consistency
        if 'precision_consistency' in self.results['detailed_tests']:
            for case_name, case_data in self.results['detailed_tests']['precision_consistency'].items():
                if 'comparisons' in case_data:
                    for comp_name, comp_data in case_data['comparisons'].items():
                        pattern_agreement = comp_data['pattern_agreement']
                        rate_similarity = comp_data['rate_similarity']
                        
                        if pattern_agreement < self.tolerances['pattern_similarity_threshold']:
                            if rate_similarity < 1.0 - self.tolerances['spike_rate_tolerance']:
                                issues.append(f"Low similarity in {case_name} {comp_name}")
                            else:
                                warnings.append(f"Pattern differences in {case_name} {comp_name}")
                        else:
                            passes.append(f"Good consistency in {case_name} {comp_name}")
        
        # Check gradient stability
        if 'gradient_stability' in self.results['detailed_tests']:
            for trial_name, trial_data in self.results['detailed_tests']['gradient_stability'].items():
                if 'comparison' in trial_data:
                    rel_error = trial_data['comparison']['relative_error']
                    if rel_error > 0.1:
                        issues.append(f"High gradient error in {trial_name}: {rel_error:.3f}")
                    elif rel_error > self.tolerances['gradient_relative_tolerance']:
                        warnings.append(f"Moderate gradient error in {trial_name}: {rel_error:.3f}")
        
        # Overall status
        if issues:
            overall_status = "❌ FAILED"
            status_color = "red"
        elif warnings:
            overall_status = "⚠️ PASSED WITH WARNINGS"
            status_color = "yellow"
        else:
            overall_status = "✅ PASSED"
            status_color = "green"
        
        # Generate recommendations
        recommendations = []
        
        if issues:
            recommendations.append("Critical numerical stability issues detected")
            recommendations.append("Consider reviewing CUDA kernel implementation")
            recommendations.append("May require higher precision for critical applications")
        
        if warnings:
            recommendations.append("Minor precision differences detected")
            recommendations.append("Acceptable for most applications")
            recommendations.append("Monitor performance in production workloads")
        
        if not issues and not warnings:
            recommendations.append("Excellent numerical stability across all tested scenarios")
            recommendations.append("All precision levels suitable for production use")
        
        # Summary
        summary = {
            'overall_status': overall_status,
            'total_tests': len(passes) + len(warnings) + len(issues),
            'passed': len(passes),
            'warnings': len(warnings),
            'failed': len(issues),
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations
        }
        
        self.results['summary'] = summary
        
        # Print summary
        print(f"Overall Status: {overall_status}")
        print(f"Tests: {summary['passed']} passed, {summary['warnings']} warnings, {summary['failed']} failed")
        
        if issues:
            print("\nCritical Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        
        return summary
    
    def save_results(self):
        """Save detailed results to JSON"""
        output_file = self.output_dir / "final_stability_results.json"
        
        # Make results JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        serializable_results = make_serializable(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Generate markdown report
        report_file = self.output_dir / "final_stability_report.md"
        with open(report_file, 'w') as f:
            f.write("# ST-BIF Numerical Stability Final Report\n\n")
            f.write(f"**Overall Status:** {self.results['summary']['overall_status']}\n\n")
            f.write(f"**Test Summary:** {self.results['summary']['passed']} passed, "
                   f"{self.results['summary']['warnings']} warnings, "
                   f"{self.results['summary']['failed']} failed\n\n")
            
            if self.results['summary']['recommendations']:
                f.write("## Recommendations\n\n")
                for rec in self.results['summary']['recommendations']:
                    f.write(f"- {rec}\n")
        
        print(f"Report saved to: {report_file}")
    
    def run_full_test_suite(self):
        """Run the complete numerical stability test suite"""
        print("ST-BIF Final Numerical Stability Test")
        print("=" * 50)
        
        self.test_precision_consistency_with_tolerance()
        self.test_boundary_sensitivity()
        self.test_gradient_stability_with_tolerance()
        
        summary = self.generate_final_assessment()
        self.save_results()
        
        return summary

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tester = ST_BIFNumericalStabilityTest(device=device)
    results = tester.run_full_test_suite()
    
    print(f"\n🏁 Test completed with status: {results['overall_status']}")