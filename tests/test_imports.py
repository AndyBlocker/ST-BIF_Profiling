"""
Import Tests - Verify all modules can be imported correctly
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCoreImports:
    """Test that core modules can be imported without errors"""
    
    def test_snn_imports(self):
        """Test SNN module imports"""
        try:
            from snn import ST_BIFNeuron_MS, MyQuan
            from snn.neurons import st_bif_neurons, if_neurons
            from snn.layers import conv, linear, normalization, pooling, quantization
            from snn.conversion import quantization as conv_quantization
        except ImportError as e:
            pytest.fail(f"Failed to import SNN modules: {e}")
    
    def test_models_imports(self):
        """Test models module imports"""
        try:
            from models import resnet
            from models.resnet import resnet18
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")
    
    def test_wrapper_imports(self):
        """Test wrapper module imports"""
        try:
            from wrapper import SNNWrapper_MS
            from wrapper.snn_wrapper import SNNWrapper_MS as SNN_MS
            from wrapper import encoding, reset, attention_conversion, base
        except ImportError as e:
            pytest.fail(f"Failed to import wrapper modules: {e}")
    
    def test_utils_imports(self):
        """Test utils module imports"""
        try:
            from utils import functions, io, misc, reset_fast
        except ImportError as e:
            pytest.fail(f"Failed to import utils modules: {e}")
    
    @pytest.mark.cuda
    def test_cuda_operator_imports(self):
        """Test CUDA operator imports"""
        try:
            from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA
        except ImportError as e:
            pytest.fail(f"Failed to import original CUDA operator: {e}")
        
        # New operator might not exist, so this is optional
        try:
            from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA as NewKernel
        except ImportError:
            pytest.skip("New CUDA operator not available (optional)")
    
    def test_pytorch_operator_imports(self):
        """Test PyTorch reference operator imports"""
        try:
            from neuron_cupy.original_operator import ST_BIFNodeATGF_MS
        except ImportError as e:
            pytest.fail(f"Failed to import PyTorch operator: {e}")


class TestImportCompatibility:
    """Test backward compatibility of imports"""
    
    def test_legacy_imports(self):
        """Test that legacy import paths still work"""
        # These should work for backward compatibility
        try:
            # Test legacy paths if they exist
            import snn
            import models 
            import wrapper
            import utils
        except ImportError as e:
            pytest.fail(f"Legacy import compatibility broken: {e}")
    
    def test_direct_class_imports(self):
        """Test direct class imports from package level"""
        try:
            from snn import ST_BIFNeuron_MS, MyQuan
            
            # Verify these are actually classes
            assert hasattr(ST_BIFNeuron_MS, '__init__')
            assert hasattr(MyQuan, '__init__')
        except ImportError as e:
            pytest.fail(f"Direct class imports failed: {e}")
    
    def test_import_without_errors(self):
        """Test that imports don't produce warnings or errors"""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import modules that should be clean
            import snn
            import models
            import wrapper
            import utils
            
            # Check if any warnings were raised
            if w:
                warning_messages = [str(warning.message) for warning in w]
                pytest.fail(f"Imports produced warnings: {warning_messages}")


class TestModuleStructure:
    """Test the structure and consistency of modules"""
    
    def test_module_has_init(self):
        """Test that all modules have proper __init__.py files"""
        module_dirs = ['snn', 'models', 'wrapper', 'utils', 'neuron_cupy']
        
        for module_dir in module_dirs:
            init_file = project_root / module_dir / '__init__.py'
            assert init_file.exists(), f"Missing __init__.py in {module_dir}"
    
    def test_submodule_structure(self):
        """Test that submodules have proper structure"""
        # Test SNN submodules
        snn_submodules = ['neurons', 'layers', 'conversion']
        for submodule in snn_submodules:
            init_file = project_root / 'snn' / submodule / '__init__.py'
            assert init_file.exists(), f"Missing __init__.py in snn.{submodule}"
    
    def test_no_circular_imports(self):
        """Test that there are no circular import dependencies"""
        # This is a basic test - more sophisticated tools like `importlib` 
        # could be used for deeper analysis
        try:
            import snn
            import models
            import wrapper
            import utils
            
            # Try importing in different orders
            del sys.modules['snn']
            del sys.modules['models'] 
            del sys.modules['wrapper']
            del sys.modules['utils']
            
            import models
            import snn
            import utils
            import wrapper
            
        except ImportError as e:
            pytest.fail(f"Possible circular import detected: {e}")


class TestOptionalDependencies:
    """Test optional dependencies and graceful degradation"""
    
    def test_cupy_availability(self):
        """Test CuPy availability for CUDA operations"""
        try:
            import cupy
            assert cupy.cuda.is_available(), "CuPy installed but CUDA not available"
        except ImportError:
            pytest.skip("CuPy not available (optional dependency)")
    
    def test_matplotlib_availability(self):
        """Test matplotlib for plotting (used in profiling)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Matplotlib not available (optional for plotting)")
    
    def test_pandas_availability(self):
        """Test pandas for data analysis (used in profiling)"""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("Pandas not available (optional for data analysis)")
    
    def test_essential_dependencies(self):
        """Test that essential dependencies are available"""
        essential = ['torch', 'numpy']
        
        for package in essential:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Essential dependency {package} not available")


class TestVersionCompatibility:
    """Test version compatibility of dependencies"""
    
    def test_torch_version(self):
        """Test PyTorch version compatibility"""
        import torch
        
        # Check minimum version requirements
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        
        assert major >= 1, f"PyTorch version too old: {version}"
        if major == 1:
            assert minor >= 8, f"PyTorch 1.x version too old: {version}"
    
    def test_cuda_availability(self):
        """Test CUDA availability if required"""
        import torch
        
        if torch.cuda.is_available():
            # If CUDA is available, test basic functionality
            device = torch.device('cuda')
            x = torch.randn(10, 10, device=device)
            y = x + 1
            assert y.device.type == device.type
        else:
            pytest.skip("CUDA not available (optional)")
    
    def test_python_version(self):
        """Test Python version compatibility"""
        import sys
        
        version = sys.version_info
        assert version.major == 3, f"Python 2 not supported"
        assert version.minor >= 8, f"Python 3.{version.minor} too old, need 3.8+"