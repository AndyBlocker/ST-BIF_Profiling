"""
Pytest configuration and shared fixtures for ST-BIF testing
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def device():
    """Test device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def test_shapes() -> List[Tuple[int, int, int]]:
    """Standard test shapes: (time_steps, batch_size, features)"""
    return [
        (10, 2, 32),      # Small batch, medium features
        (5, 16, 64),      # Medium batch, large features
        (20, 1, 16),      # Long sequence, single batch
        (15, 8, 128),     # Medium sequence, large features
        (32, 4, 256),     # Large sequence, large features
        (8, 32, 64),      # Short sequence, large batch
    ]

@pytest.fixture
def threshold_values(device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard threshold values: (v_th, T_max, T_min)"""
    return (
        torch.tensor(0.1, device=device),
        torch.tensor(4.0, device=device),
        torch.tensor(-4.0, device=device)
    )

@pytest.fixture
def tolerance_config():
    """Tolerance configuration for different precisions"""
    return {
        'fp64': {'atol': 1e-7, 'rtol': 1e-5},
        'fp32': {'atol': 1e-5, 'rtol': 1e-4},
        'fp16': {'atol': 1e-3, 'rtol': 1e-2}
    }

@pytest.fixture
def dtype_map():
    """Mapping from precision names to torch dtypes"""
    return {
        'fp64': torch.float64,
        'fp32': torch.float32,
        'fp16': torch.float16
    }

def generate_input_data(shape: Tuple[int, int, int], device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Generate realistic input data for neural network testing"""
    time_steps, batch_size, features = shape
    # Generate values with realistic distribution for neuron inputs
    x = (torch.randn(time_steps, batch_size, features, device=device, dtype=dtype) * 0.5)
    x.requires_grad = True
    return x

def assert_tensor_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-6, test_name=""):
    """Enhanced tensor comparison with detailed error reporting"""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(a - b)).item()
        mean_diff = torch.mean(torch.abs(a - b)).item()
        diff_locations = torch.where(torch.abs(a - b) > atol)
        
        error_msg = (
            f"Tensor comparison failed for {test_name}\n"
            f"Max difference: {max_diff:.6e}\n"
            f"Mean difference: {mean_diff:.6e}\n"
            f"Tolerance: rtol={rtol}, atol={atol}\n"
            f"Number of differing elements: {len(diff_locations[0])}\n"
            f"Total elements: {a.numel()}\n"
            f"Difference ratio: {len(diff_locations[0])/a.numel():.2%}\n"
        )
        
        if len(diff_locations[0]) > 0:
            error_msg += f"First few differences:\n"
            error_msg += f"  a: {a[diff_locations][:5].detach().cpu().numpy()}\n"
            error_msg += f"  b: {b[diff_locations][:5].detach().cpu().numpy()}\n"
        
        pytest.fail(error_msg)

# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "equivalence: mark test as equivalence test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "regression: mark test as regression test")

def pytest_collection_modifyitems(config, items):
    """Automatically skip CUDA tests if CUDA is not available"""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)