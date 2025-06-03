# ST-BIF SNN Profiling Usage Guide

Quick reference for profiling the ST-BIF Spiking Neural Network framework.

## Quick Commands

### 🚀 Quick Development Profiling
```bash
# Fast benchmark for development workflow
./profile/scripts/quick_profile.sh
```

### 📊 Comprehensive Profiling
```bash
# Run all profiling methods
./profile/scripts/run_all_profiles.sh

# Quick mode (faster, smaller batch)
./profile/scripts/run_all_profiles.sh --quick

# Thorough mode (slower, more detailed)
./profile/scripts/run_all_profiles.sh --thorough
```

### 🔍 Individual Profiling Methods

#### PyTorch Profiler
```bash
# Basic profiling with torch.profiler
python profile/scripts/snn_profiler.py --method torch --steps 100

# Memory-focused profiling
python profile/scripts/snn_profiler.py --method memory --steps 50

# Throughput benchmarking
python profile/scripts/snn_profiler.py --method benchmark --steps 100

# All PyTorch profiling methods
python profile/scripts/snn_profiler.py --method all
```

#### NVIDIA Nsight Systems
```bash
# GPU timeline profiling
./profile/scripts/nsys_profile.sh

# Custom configuration
./profile/scripts/nsys_profile.sh --batch-size 32 --steps 100
```

#### NVIDIA Nsight Compute
```bash
# Detailed kernel profiling
./profile/scripts/ncu_profile.sh

# Focus on SNN kernels only
./profile/scripts/ncu_profile.sh --snn-kernels

# Custom kernel filtering
./profile/scripts/ncu_profile.sh --kernel-regex ".*conv.*"
```

## Analysis Workflow

### 1. Quick Check During Development
```bash
./profile/scripts/quick_profile.sh
```
- Takes ~1 minute
- Provides basic throughput metrics
- Good for checking regressions

### 2. Forward/Backward Analysis
```bash
python profile/scripts/snn_profiler.py --method all --steps 100
```
- Detailed breakdown of forward vs backward pass
- Memory usage patterns
- Chrome trace visualization

### 3. GPU Optimization
```bash
./profile/scripts/nsys_profile.sh --steps 100
```
- GPU utilization timeline
- Memory transfer analysis
- Kernel launch patterns

### 4. Kernel-Level Optimization
```bash
./profile/scripts/ncu_profile.sh --snn-kernels --steps 20
```
- Detailed kernel performance
- Memory bandwidth utilization
- Register usage optimization

## Key Metrics to Watch

### SNN Performance Indicators
- **Forward Time**: Should be ~10-20% of total
- **Backward Time**: Typically 70-80% of total  
- **Memory Usage**: Peak allocation during backward pass
- **ST-BIF Operations**: Custom neuron kernel efficiency

### Typical Performance Profile
```
Forward Pass:   ~8ms  (10-15%)
Backward Pass: ~66ms  (80-85%)
Total:         ~74ms
Throughput:    ~216 samples/sec (batch_size=16)
```

## Common Optimizations

### 1. Batch Size Tuning
```bash
# Test different batch sizes
python profile/scripts/snn_profiler.py --batch-size 8 --method benchmark
python profile/scripts/snn_profiler.py --batch-size 16 --method benchmark  
python profile/scripts/snn_profiler.py --batch-size 32 --method benchmark
```

### 2. Time Step Analysis
```bash
# Compare different temporal resolutions
python profile/scripts/snn_profiler.py --time-steps 4 --method benchmark
python profile/scripts/snn_profiler.py --time-steps 8 --method benchmark
python profile/scripts/snn_profiler.py --time-steps 16 --method benchmark
```

### 3. Quantization Level Impact
```bash
# Test quantization effects
python profile/scripts/snn_profiler.py --level 4 --method benchmark
python profile/scripts/snn_profiler.py --level 8 --method benchmark
python profile/scripts/snn_profiler.py --level 16 --method benchmark
```

## Viewing Results

### PyTorch Profiler Results
```bash
# Chrome trace (interactive timeline)
# Open profile/outputs/torch_profile_*.json in Chrome at chrome://tracing/

# Text reports
cat profile/outputs/torch_profile_*.txt
cat profile/outputs/benchmark_*.txt
```

### NVIDIA Tools Results
```bash
# Nsight Systems
nsys-ui profile/outputs/nsys_profile_*.nsys-rep

# Nsight Compute  
ncu-ui profile/outputs/ncu_profile_*.ncu-rep
```

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python profile/scripts/snn_profiler.py --batch-size 8

# CPU profiling if CUDA OOM
python profile/scripts/snn_profiler.py --device cpu
```

### Slow Profiling
```bash
# Use quick mode
./profile/scripts/run_all_profiles.sh --quick

# Skip detailed kernel analysis
python profile/scripts/snn_profiler.py --method torch
```

### Missing NVIDIA Tools
```bash
# Check installation
which nsys ncu

# Install via conda
conda install -c nvidia nsight-systems nsight-compute

# Or download from NVIDIA website
```

## Performance Baselines

### Expected Results (CUDA, ResNet18, CIFAR-10)
| Batch Size | Forward (ms) | Backward (ms) | Total (ms) | Throughput (samples/sec) |
|------------|--------------|---------------|------------|--------------------------|
| 8          | ~4           | ~33           | ~37        | ~216                     |
| 16         | ~8           | ~66           | ~74        | ~216                     |
| 32         | ~15          | ~130          | ~145       | ~220                     |

*Results may vary based on GPU model and system configuration*

## Integration Examples

### CI/CD Performance Testing
```bash
# Add to your CI pipeline
./profile/scripts/quick_profile.sh > performance_report.txt
# Check for performance regressions
```

### Development Workflow
```bash
# Before making changes
./profile/scripts/quick_profile.sh > baseline.txt

# After making changes  
./profile/scripts/quick_profile.sh > modified.txt

# Compare results
diff baseline.txt modified.txt
```

### Detailed Analysis Session
```bash
# Comprehensive analysis
./profile/scripts/run_all_profiles.sh --thorough

# Focus on bottlenecks found
./profile/scripts/ncu_profile.sh --kernel-regex ".*identified_bottleneck.*"
```