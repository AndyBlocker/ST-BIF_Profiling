# ST-BIF SNN Performance Profiling Suite

This directory contains comprehensive profiling tools for analyzing the performance characteristics of the ST-BIF Spiking Neural Network framework, with focus on forward and backward pass bottlenecks.

## Overview

The profiling suite supports three major profiling methods:

1. **PyTorch Profiler** (`torch.profiler`) - PyTorch native profiling
2. **NVIDIA Nsight Systems** (`nsys`) - GPU timeline and system-level analysis
3. **NVIDIA Nsight Compute** (`ncu`) - Detailed kernel-level analysis

## Directory Structure

```
profile/
├── README.md                    # This file (overview and quick start)
├── USAGE.md                     # Detailed usage guide
├── NSYS_PROFILING_GUIDE.md     # NSYS-specific analysis guide
├── analysis_report.md           # Analysis report template
├── scripts/                     # Profiling scripts
│   ├── nsys_snn_profiling.py   # NSYS-specific SNN analysis
│   └── run_nsys_profiling.sh   # NSYS execution script
├── configs/                     # Configuration files
│   └── profile_config.yaml     # Profiling parameters
└── outputs/                     # Profiling results (auto-created)
    └── nsys_results/           # NSYS-specific results
```

## Quick Start

### NSYS Profiling (唯一有效的分析工具)
```bash
# NSYS详细GPU分析（推荐）
cd profile/scripts
./run_nsys_profiling.sh

# 或者手动运行
cd profile/scripts  
python nsys_snn_profiling.py
```

### Individual Profiling Methods

#### 1. PyTorch Profiler
```bash
# Basic PyTorch profiling
python profile/scripts/snn_profiler.py --method torch

# Memory-focused profiling
python profile/scripts/snn_profiler.py --method memory --steps 50

# Throughput benchmarking
python profile/scripts/snn_profiler.py --method benchmark --steps 100

# All PyTorch profiling methods
python profile/scripts/snn_profiler.py --method all
```

#### 2. NVIDIA Nsight Systems (NVTX)
```bash
# Detailed SNN profiling with NVTX markers
./profile/scripts/run_nsys_profiling.sh

# Alternative basic nsys profiling
./profile/scripts/nsys_profile.sh --batch-size 64 --steps 200
```

#### 3. NVIDIA Nsight Compute
```bash
# Detailed kernel profiling
./profile/scripts/ncu_profile.sh

# Focus on SNN-specific kernels
./profile/scripts/ncu_profile.sh --snn-kernels

# Custom kernel filtering
./profile/scripts/ncu_profile.sh --kernel-regex ".*conv.*"
```

## Configuration Options

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--batch-size` | Batch size for profiling | 32 |
| `--steps` | Number of profiling steps | 100 |
| `--warmup` | Warmup steps before profiling | 20 |
| `--time-steps` | SNN temporal time steps | 8 |
| `--level` | Quantization level | 8 |
| `--device` | Device (cuda/cpu) | cuda |

### Profiling Modes

#### Quick Mode (`--quick`)
- Batch Size: 16
- Steps: 50
- Warmup: 10
- Best for: Rapid iteration and debugging

#### Thorough Mode (`--thorough`)
- Batch Size: 64
- Steps: 200
- Warmup: 50
- Best for: Comprehensive analysis

## Output Files

Profiling results are saved in `profile/outputs/` with timestamps:

### PyTorch Profiler
- `torch_profile_<timestamp>.json` - Chrome trace for visualization
- `torch_profile_<timestamp>.txt` - Detailed text report
- `memory_profile_<timestamp>.txt` - Memory usage analysis
- `benchmark_<timestamp>.txt` - Throughput results

### NVIDIA Nsight Systems
- `nsys_profile_<timestamp>.nsys-rep` - Main report file
- `nsys_profile_<timestamp>.sqlite` - Database export
- `nsys_summary_<timestamp>.txt` - Text summary

### NVIDIA Nsight Compute
- `ncu_profile_<timestamp>.ncu-rep` - Detailed kernel report
- `ncu_summary_<timestamp>.txt` - Summary analysis

## Analysis Workflow

### 1. Start with PyTorch Profiler
```bash
python profile/scripts/snn_profiler.py --method all --steps 100
```

**What to look for:**
- Top operations by CPU/CUDA time
- Memory allocation patterns
- Forward vs backward pass breakdown
- ST-BIF neuron operation costs

**View results:**
- Open `*.json` files in Chrome at `chrome://tracing/`
- Review `*.txt` files for detailed statistics

### 2. GPU Timeline Analysis (Nsight Systems)
```bash
./profile/scripts/nsys_profile.sh --steps 100
```

**What to look for:**
- GPU utilization over time
- Memory transfer patterns
- Kernel launch overhead
- Synchronization bottlenecks

**View results:**
```bash
nsys-ui profile/outputs/nsys_profile_*.nsys-rep
```

### 3. Kernel-Level Optimization (Nsight Compute)
```bash
./profile/scripts/ncu_profile.sh --snn-kernels --steps 20
```

**What to look for:**
- Compute throughput utilization
- Memory bandwidth efficiency
- Warp execution efficiency
- Register usage patterns

**View results:**
```bash
ncu-ui profile/outputs/ncu_profile_*.ncu-rep
```

## Key Metrics to Monitor

### SNN-Specific Metrics
- **ST-BIF Neuron Operations**: Forward/backward pass timing
- **Temporal Encoding**: Time step processing efficiency
- **Spike Generation**: Threshold comparison operations
- **Memory Patterns**: Sparse spike data handling

### General Performance Metrics
- **Forward Pass Time**: Input → spike output latency
- **Backward Pass Time**: Gradient computation through time
- **Memory Usage**: Peak and average allocation
- **Throughput**: Samples processed per second
- **GPU Utilization**: Kernel occupancy and efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python profile/scripts/snn_profiler.py --batch-size 16
   ```

2. **nsys/ncu Not Found**
   ```bash
   # Install NVIDIA tools or add to PATH
   export PATH=/usr/local/cuda/bin:$PATH
   ```

3. **Slow NCU Profiling**
   ```bash
   # Use fewer steps for detailed analysis
   ./profile/scripts/ncu_profile.sh --steps 10
   ```

4. **No CUDA Device**
   ```bash
   # Use CPU profiling
   python profile/scripts/snn_profiler.py --device cpu
   ```

### Performance Tips

1. **Warmup Properly**: Use sufficient warmup steps to stabilize CUDA kernels
2. **Focus Analysis**: Use kernel filtering for targeted optimization
3. **Batch Size**: Balance between realistic workloads and memory constraints
4. **Multiple Runs**: Run profiling multiple times to ensure consistency

## Advanced Usage

### Custom Kernel Analysis
```bash
# Profile specific kernel patterns
./profile/scripts/ncu_profile.sh --kernel-regex ".*st_bif.*|.*spiking.*"

# Profile convolution operations
./profile/scripts/ncu_profile.sh --kernel-regex ".*conv.*|.*gemm.*"
```

### Memory-Focused Profiling
```bash
# Large batch size for memory analysis
python profile/scripts/snn_profiler.py --method memory --batch-size 128
```

### Comparative Analysis
```bash
# Profile different configurations
python profile/scripts/snn_profiler.py --time-steps 4 --level 4
python profile/scripts/snn_profiler.py --time-steps 8 --level 8
python profile/scripts/snn_profiler.py --time-steps 16 --level 16
```

## Integration with Development

### Continuous Profiling
Add profiling to your development workflow:

```bash
# Quick check during development
./profile/scripts/run_all_profiles.sh --quick

# Before major commits
./profile/scripts/run_all_profiles.sh --thorough
```

### Performance Regression Detection
Compare profiling results across different code versions to detect performance regressions.

## Contributing

When adding new profiling capabilities:

1. Follow the existing script structure
2. Add appropriate documentation
3. Include error handling and validation
4. Test with different configurations
5. Update this README

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)