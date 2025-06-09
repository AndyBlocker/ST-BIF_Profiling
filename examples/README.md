# ST-BIF Framework Examples

This directory contains examples demonstrating the usage of the ST-BIF (Spike Threshold Bifurcation) modular framework for converting artificial neural networks to spiking neural networks.

## Available Examples

### `ann_to_snn_conversion.py`

**Complete ANN to SNN Conversion Pipeline**

This example demonstrates the full conversion pipeline from artificial neural networks (ANNs) to spiking neural networks (SNNs) using the modular ST-BIF framework.

**Features:**
- Load pre-trained ResNet18 models on CIFAR-10
- Convert ANN to quantized ANN (QANN) with learnable quantizers
- Convert QANN to SNN with ST-BIF neurons
- Performance comparison across all model types
- Detailed timing and accuracy analysis

**Usage:**
```bash
# Basic usage with default settings
python examples/ann_to_snn_conversion.py

# Custom batch size
python examples/ann_to_snn_conversion.py --batch-size 64

# Quiet mode for benchmarking
python examples/ann_to_snn_conversion.py --quiet

# Custom model paths
python examples/ann_to_snn_conversion.py \
    --ann-path path/to/ann_model.pth \
    --qann-path path/to/qann_model.pth
```

**Expected Output:**
```
🧪 ANN TO SNN CONVERSION EXAMPLE
Framework: ST-BIF Modular Framework
Device: CUDA
Batch Size: 128

📊 CONVERSION PIPELINE RESULTS
Model      Accuracy     Time (ms)    Speed      Acc Change  
----------------------------------------------------------------------
ANN        86.74%     0.025       baseline   baseline    
QANN       85.17%     0.023       1.07x      -1.57%      
SNN        85.12%     0.055       0.45x      -1.62%      
----------------------------------------------------------------------

🔍 Conversion Analysis:
  ANN → QANN: 86.74% → 85.17% (-1.57%), 1.07x speed
  QANN → SNN: 85.17% → 85.12% (-0.05%), 0.43x speed
  Overall ANN → SNN: 86.74% → 85.12% (-1.62%), 0.45x speed
```

## Framework Architecture

The examples demonstrate the modular architecture:

```
ST-BIF_Profiling/
├── snn/               # Core SNN framework (neurons, layers, conversion)
├── models/            # Neural network models
├── wrapper/           # SNN wrapper classes
├── utils/             # Framework utilities
└── examples/          # Usage examples (this directory)
```

## Model Requirements

The examples expect pre-trained model checkpoints:
- `checkpoints/resnet/best_ANN.pth` - Trained ResNet18 on CIFAR-10
- `checkpoints/resnet/best_QANN.pth` - Quantized version with learned quantizers

If these files are not available, the examples will attempt to convert from the ANN model or may fail gracefully.

## Key Components Demonstrated

### 1. Model Conversion
```python
from snn.conversion import myquan_replace_resnet
from wrapper import SNNWrapper_MS

# Convert ANN to QANN
myquan_replace_resnet(model, level=8, weight_bit=32)

# Convert QANN to SNN
snn_model = SNNWrapper_MS(
    ann_model=qann_model,
    time_step=8,
    level=8,
    neuron_type="ST-BIF"
)
```

### 2. ST-BIF Neurons
The framework uses ST-BIF (Spike Threshold Bifurcation) neurons that provide:
- Learnable thresholds
- Multi-step temporal processing
- CUDA acceleration support
- Energy-efficient spike-based computation

### 3. Temporal Encoding
- 8 time steps for temporal spike encoding
- Analog-to-spike conversion
- Accumulation across time steps

## Performance Characteristics

**Typical Results:**
- **Accuracy**: ~85% on CIFAR-10 (minimal degradation from ANN)
- **Speed**: 0.4-0.5x of ANN inference speed
- **Energy**: Significantly lower due to sparse spiking activity
- **Memory**: Comparable to quantized models

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size with `--batch-size 32`
2. **Missing Model Files**: Check paths in `checkpoints/resnet/`
3. **Import Errors**: Ensure you're running from project root directory

**Getting Help:**
- Check the main project README for setup instructions
- Verify CUDA installation for GPU acceleration
- Ensure all dependencies are installed

## Contributing

To add new examples:
1. Create a new `.py` file in this directory
2. Follow the existing code structure and documentation style
3. Add usage instructions to this README
4. Test with different configurations