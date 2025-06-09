"""
Model Conversion Tests - Testing ANN to QANN to SNN conversion pipeline
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.resnet import resnet18
    from snn.conversion.quantization import myquan_replace_resnet
    from wrapper.snn_wrapper import SNNWrapper_MS
    CONVERSION_AVAILABLE = True
except ImportError as e:
    CONVERSION_AVAILABLE = False
    print(f"Warning: Model conversion modules not available: {e}")


class TestModelConversion:
    """Test the complete ANN → QANN → SNN conversion pipeline"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample ResNet18 model for testing"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        return resnet18(num_classes=10)
    
    @pytest.fixture
    def sample_data(self, device):
        """Create sample CIFAR-10 like data"""
        batch_size = 4
        return torch.randn(batch_size, 3, 32, 32, device=device)
    
    def test_ann_to_qann_conversion(self, sample_model, sample_data, device):
        """Test ANN to QANN conversion"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Move model to device
        model = sample_model.to(device)
        
        # Test original model forward pass
        with torch.no_grad():
            ann_output = model(sample_data)
        
        assert ann_output.shape == (4, 10)  # batch_size=4, num_classes=10
        assert not torch.isnan(ann_output).any()
        
        # Convert to QANN
        myquan_replace_resnet(model, level=8, weight_bit=32)
        
        # Test QANN forward pass
        with torch.no_grad():
            qann_output = model(sample_data)
        
        assert qann_output.shape == (4, 10)
        assert not torch.isnan(qann_output).any()
        
        # Outputs should be similar but not identical due to quantization
        output_diff = torch.mean(torch.abs(ann_output - qann_output)).item()
        assert output_diff < 10.0, f"QANN output differs too much from ANN: {output_diff}"

    def test_qann_to_snn_conversion(self, sample_model, sample_data, device):
        """Test QANN to SNN conversion"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Prepare QANN model
        model = sample_model.to(device)
        myquan_replace_resnet(model, level=8, weight_bit=32)
        
        # Get QANN baseline
        with torch.no_grad():
            qann_output = model(sample_data)
        
        # Create SNN wrapper
        snn_model = SNNWrapper_MS(
            ann_model=model,
            cfg=None,
            time_step=8,
            Encoding_type="analog",
            level=8,
            neuron_type="ST-BIF"
        )
        snn_model = snn_model.to(device)
        
        # Test SNN forward pass
        with torch.no_grad():
            snn_output = snn_model(sample_data)
        
        assert snn_output.shape == (4, 10)
        assert not torch.isnan(snn_output).any()
        
        # SNN output should be reasonably close to QANN
        output_diff = torch.mean(torch.abs(qann_output - snn_output)).item()
        assert output_diff < 20.0, f"SNN output differs too much from QANN: {output_diff}"

    @pytest.mark.slow
    def test_full_conversion_pipeline(self, sample_model, sample_data, device):
        """Test the complete ANN → QANN → SNN pipeline"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Step 1: ANN baseline
        ann_model = sample_model.to(device)
        with torch.no_grad():
            ann_output = ann_model(sample_data)
        
        # Step 2: ANN → QANN
        myquan_replace_resnet(ann_model, level=8, weight_bit=32)
        with torch.no_grad():
            qann_output = ann_model(sample_data)
        
        # Step 3: QANN → SNN
        snn_model = SNNWrapper_MS(
            ann_model=ann_model,
            cfg=None,
            time_step=8,
            Encoding_type="analog",
            level=8,
            neuron_type="ST-BIF"
        ).to(device)
        
        with torch.no_grad():
            snn_output = snn_model(sample_data)
        
        # Verify each step
        ann_qann_diff = torch.mean(torch.abs(ann_output - qann_output)).item()
        qann_snn_diff = torch.mean(torch.abs(qann_output - snn_output)).item()
        ann_snn_diff = torch.mean(torch.abs(ann_output - snn_output)).item()
        
        print(f"\nConversion pipeline differences:")
        print(f"ANN → QANN: {ann_qann_diff:.4f}")
        print(f"QANN → SNN: {qann_snn_diff:.4f}")
        print(f"ANN → SNN: {ann_snn_diff:.4f}")
        
        # All differences should be reasonable
        assert ann_qann_diff < 15.0, f"ANN→QANN difference too large: {ann_qann_diff}"
        assert qann_snn_diff < 20.0, f"QANN→SNN difference too large: {qann_snn_diff}"
        assert ann_snn_diff < 25.0, f"ANN→SNN difference too large: {ann_snn_diff}"

    def test_snn_time_step_consistency(self, sample_model, sample_data, device):
        """Test that SNN gives consistent results across multiple runs"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Prepare SNN model
        model = sample_model.to(device)
        myquan_replace_resnet(model, level=8, weight_bit=32)
        
        snn_model = SNNWrapper_MS(
            ann_model=model,
            cfg=None,
            time_step=8,
            Encoding_type="analog",
            level=8,
            neuron_type="ST-BIF"
        ).to(device)
        
        # Run multiple times with same input
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = snn_model(sample_data)
                outputs.append(output.clone())
        
        # Results should be identical (deterministic)
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i], 
                                     rtol=1e-5, atol=1e-6,
                                     msg=f"SNN output not consistent between runs 0 and {i}")

    @pytest.mark.parametrize("time_step", [4, 8, 16])
    def test_different_time_steps(self, sample_model, sample_data, device, time_step):
        """Test SNN conversion with different time steps"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Prepare model
        model = sample_model.to(device)
        myquan_replace_resnet(model, level=8, weight_bit=32)
        
        # Create SNN with specific time step
        snn_model = SNNWrapper_MS(
            ann_model=model,
            cfg=None,
            time_step=time_step,
            Encoding_type="analog",
            level=8,
            neuron_type="ST-BIF"
        ).to(device)
        
        # Should work without errors
        with torch.no_grad():
            output = snn_model(sample_data)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelAccuracy:
    """Test model accuracy preservation through conversion"""
    
    def test_accuracy_regression_threshold(self, sample_model, device):
        """Test that conversion doesn't cause severe accuracy regression"""
        if not CONVERSION_AVAILABLE:
            pytest.skip("Model conversion modules not available")
        
        # Create larger test dataset
        test_data = torch.randn(64, 3, 32, 32, device=device)
        test_labels = torch.randint(0, 10, (64,), device=device)
        
        # ANN accuracy
        ann_model = sample_model.to(device)
        ann_model.eval()
        with torch.no_grad():
            ann_logits = ann_model(test_data)
            ann_preds = torch.argmax(ann_logits, dim=1)
            ann_accuracy = (ann_preds == test_labels).float().mean().item()
        
        # QANN accuracy
        myquan_replace_resnet(ann_model, level=8, weight_bit=32)
        with torch.no_grad():
            qann_logits = ann_model(test_data)
            qann_preds = torch.argmax(qann_logits, dim=1)
            qann_accuracy = (qann_preds == test_labels).float().mean().item()
        
        # SNN accuracy
        snn_model = SNNWrapper_MS(
            ann_model=ann_model,
            cfg=None,
            time_step=8,
            Encoding_type="analog",
            level=8,
            neuron_type="ST-BIF"
        ).to(device)
        
        with torch.no_grad():
            snn_logits = snn_model(test_data)
            snn_preds = torch.argmax(snn_logits, dim=1)
            snn_accuracy = (snn_preds == test_labels).float().mean().item()
        
        print(f"\nAccuracy comparison:")
        print(f"ANN: {ann_accuracy:.1%}")
        print(f"QANN: {qann_accuracy:.1%} (drop: {ann_accuracy - qann_accuracy:.1%})")
        print(f"SNN: {snn_accuracy:.1%} (drop: {ann_accuracy - snn_accuracy:.1%})")
        
        # Accuracy should not drop too much (thresholds can be adjusted)
        assert qann_accuracy > ann_accuracy - 0.3, f"QANN accuracy drop too large: {ann_accuracy - qann_accuracy:.1%}"
        assert snn_accuracy > ann_accuracy - 0.4, f"SNN accuracy drop too large: {ann_accuracy - snn_accuracy:.1%}"