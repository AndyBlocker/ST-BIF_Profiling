"""
ST-BIF Profiling: Spiking Neural Network Framework
=================================================

A comprehensive framework for spiking neural networks with quantization support.
Implements the ST-BIF (Spike Threshold Bifurcation) profiling approach.

Architecture:
- layer/snn/: Core SNN layers and neurons
- conversion/: Model conversion utilities (ANN->QANN->SNN) 
- wrapper/: Model wrapper classes and preprocessing
- utils/: Framework-level utilities and helper functions

Usage:
    # Import SNN layers
    from layer.snn import ST_BIFNeuron_MS, MyQuan, LLConv2d_MS
    
    # Import conversion tools
    from conversion import myquan_replace_resnet
    
    # Import wrappers
    from wrapper import SNNWrapper_MS
"""

__version__ = "0.1.0"
__author__ = "ST-BIF Research Team"
__description__ = "ST-BIF Profiling: Spiking Neural Network Framework"

# Framework metadata
FRAMEWORK_INFO = {
    "name": "ST-BIF Profiling",
    "version": __version__,
    "description": __description__,
    "components": {
        "layer.snn": "Core SNN layers and neurons",
        "conversion": "Model conversion utilities",
        "wrapper": "Model wrapper classes", 
        "utils": "Framework utilities"
    }
}

def get_framework_info():
    """Return framework information"""
    return FRAMEWORK_INFO

def check_framework_integrity():
    """Check if all framework components are available"""
    import os
    components = ['layer/snn', 'conversion', 'wrapper', 'utils']
    missing = []
    
    for component in components:
        if not os.path.exists(component):
            missing.append(component)
    
    if missing:
        print(f"❌ Missing framework components: {missing}")
        return False
    else:
        print("✅ All framework components available")
        return True

# Run integrity check when imported
if __name__ != '__main__':
    check_framework_integrity()