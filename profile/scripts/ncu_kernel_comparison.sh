#!/bin/bash

# ST-BIF CUDA Kernel NCU Comparison Script
# ========================================
# This script uses NVIDIA Nsight Compute (NCU) to perform detailed
# kernel-level analysis and comparison between original and new CUDA kernels

set -e

echo "🔬 ST-BIF CUDA Kernel NCU Analysis"
echo "=================================="

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../outputs/nsys_results"
BATCH_SIZE=32
TIME_STEPS=8
FEATURE_SIZE=512

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Time steps: $TIME_STEPS" 
echo "  Feature size: $FEATURE_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ ncu (NVIDIA Nsight Compute) not found."
    echo "Please install NVIDIA Nsight Compute and ensure it's in your PATH."
    echo ""
    echo "Alternative: Run basic profiling without NCU..."
    python cuda_kernel_profiler.py
    exit 0
fi

echo "📊 Running NCU kernel analysis..."
echo ""

# Function to run NCU analysis
run_ncu_analysis() {
    local kernel_type="$1"
    local output_prefix="$2"
    
    echo "Analyzing $kernel_type kernel..."
    
    # Create a temporary Python script for this specific kernel
    cat > "temp_${kernel_type}_test.py" << EOF
#!/usr/bin/env python3
import sys
sys.path.append('../..')

import torch
import torch.cuda.nvtx as nvtx
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

def test_${kernel_type}_kernel():
    device = 'cuda'
    batch_size = $BATCH_SIZE
    time_steps = $TIME_STEPS
    feature_size = $FEATURE_SIZE
    
    # Generate test data
    torch.manual_seed(42)
    x_seq = torch.randn(time_steps * batch_size, feature_size, device=device)
    
    # Create neuron
    neuron = ST_BIFNeuron_MS(
        q_threshold=torch.tensor(1.0),
        level=8,
        sym=True,
        first_neuron=True
    ).to(device)
    neuron.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            neuron._reset_states()
            _ = neuron(x_seq)
    
    torch.cuda.synchronize()
    
    # Main computation with NVTX markers
    nvtx.range_push("${kernel_type}_Kernel_Analysis")
    
    with torch.no_grad():
        neuron._reset_states()
        output = neuron(x_seq)
    
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    print(f"${kernel_type} kernel completed. Output shape: {output.shape}")

if __name__ == "__main__":
    test_${kernel_type}_kernel()
EOF
    
    # Run NCU with comprehensive metrics
    echo "  Running detailed NCU analysis (this may take several minutes)..."
    
    ncu \
        --target-processes all \
        --kernel-regex ".*st_bif.*|.*ST_BIF.*|.*forward.*|.*backward.*" \
        --launch-skip-before-match 0 \
        --launch-count 10 \
        --clock-control none \
        --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,dram__bytes.sum,l1tex__t_bytes.sum,sm__sass_thread_inst_executed.sum,sm__inst_executed.sum,smsp__sass_thread_inst_executed.sum,sm__pipe_tensor_cycles_active.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv \
        --log-file "${OUTPUT_DIR}/ncu_${output_prefix}_${TIMESTAMP}.log" \
        --export "${OUTPUT_DIR}/ncu_${output_prefix}_${TIMESTAMP}" \
        python "temp_${kernel_type}_test.py" \
        > "${OUTPUT_DIR}/ncu_${output_prefix}_stdout_${TIMESTAMP}.txt" 2>&1
    
    # Cleanup temporary file
    rm "temp_${kernel_type}_test.py"
    
    echo "  ✓ $kernel_type analysis completed"
}

# Function to run basic kernel comparison without NCU
run_basic_comparison() {
    echo "🔄 Running basic kernel comparison..."
    python cuda_kernel_profiler.py
    echo "  ✓ Basic comparison completed"
}

# Function to run lightweight NCU analysis
run_lightweight_ncu() {
    echo "⚡ Running lightweight NCU analysis..."
    
    # Create a single test script
    cat > "temp_kernel_test.py" << EOF
#!/usr/bin/env python3
import sys
sys.path.append('../..')

import torch
import torch.cuda.nvtx as nvtx
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS

def test_kernels():
    device = 'cuda'
    batch_size = $BATCH_SIZE
    time_steps = $TIME_STEPS
    feature_size = $FEATURE_SIZE
    
    # Generate test data
    torch.manual_seed(42)
    x_seq = torch.randn(time_steps * batch_size, feature_size, device=device)
    
    # Create neuron
    neuron = ST_BIFNeuron_MS(
        q_threshold=torch.tensor(1.0),
        level=8,
        sym=True,
        first_neuron=True
    ).to(device)
    neuron.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            neuron._reset_states()
            _ = neuron(x_seq)
    
    torch.cuda.synchronize()
    
    # Multiple runs for analysis
    nvtx.range_push("ST_BIF_Kernel_Runs")
    
    for i in range(5):
        nvtx.range_push(f"ST_BIF_Run_{i}")
        with torch.no_grad():
            neuron._reset_states()
            output = neuron(x_seq)
        torch.cuda.synchronize()
        nvtx.range_pop()
    
    nvtx.range_pop()
    
    print(f"Kernel analysis completed. Output shape: {output.shape}")

if __name__ == "__main__":
    test_kernels()
EOF
    
    # Run lightweight NCU analysis
    ncu \
        --target-processes all \
        --kernel-regex ".*st_bif.*|.*ST_BIF.*|.*forward.*|.*backward.*" \
        --launch-count 5 \
        --clock-control none \
        --metrics dram__bytes.sum,sm__cycles_elapsed.avg,sm__inst_executed.sum \
        --csv \
        --export "${OUTPUT_DIR}/ncu_st_bif_lightweight_${TIMESTAMP}" \
        python temp_kernel_test.py \
        > "${OUTPUT_DIR}/ncu_lightweight_stdout_${TIMESTAMP}.txt" 2>&1
    
    # Cleanup
    rm temp_kernel_test.py
    
    echo "  ✓ Lightweight NCU analysis completed"
}

# Main execution logic
echo "🎯 Choose analysis mode:"
echo "  1. Full NCU analysis (detailed but slow)"
echo "  2. Lightweight NCU analysis (faster)" 
echo "  3. Basic Python profiling only"
echo ""

# Auto-select lightweight mode for automated runs
ANALYSIS_MODE="2"

case "$ANALYSIS_MODE" in
    "1")
        echo "Running full NCU analysis..."
        run_ncu_analysis "original" "original"
        if [ -f "../neuron_cupy/cuda_operator_new.py" ]; then
            run_ncu_analysis "new" "new"
        fi
        ;;
    "2")
        echo "Running lightweight NCU analysis..."
        run_lightweight_ncu
        ;;
    "3")
        echo "Running basic profiling only..."
        run_basic_comparison
        ;;
    *)
        echo "Invalid choice, running basic profiling..."
        run_basic_comparison
        ;;
esac

# Generate summary report
echo ""
echo "📋 Generating analysis summary..."

cat > "${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt" << EOF
ST-BIF CUDA Kernel NCU Analysis Summary
======================================

Analysis Configuration:
  Timestamp: $TIMESTAMP
  Batch size: $BATCH_SIZE
  Time steps: $TIME_STEPS
  Feature size: $FEATURE_SIZE
  Analysis mode: $ANALYSIS_MODE

Generated Files:
EOF

# List generated files
ls -la "${OUTPUT_DIR}"/ncu_*${TIMESTAMP}* | while read -r line; do
    filename=$(echo "$line" | awk '{print $9}')
    if [ -n "$filename" ]; then
        echo "  - $(basename "$filename")" >> "${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt"
    fi
done

cat >> "${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt" << EOF

Analysis Notes:
- Use 'ncu-ui' to open .ncu-rep files for detailed GUI analysis
- CSV files contain raw metrics data
- Log files contain console output from NCU runs

Key Metrics to Examine:
1. SM Cycles Elapsed: Overall kernel execution time
2. DRAM Throughput: Memory bandwidth utilization  
3. Instruction Throughput: Compute utilization
4. Tensor Core Usage: Specialized compute usage
5. Memory Access Patterns: L1/L2 cache efficiency

NCU GUI Commands:
  ncu-ui ${OUTPUT_DIR}/ncu_*_${TIMESTAMP}.ncu-rep

EOF

echo "✅ Analysis completed!"
echo ""
echo "📁 Results saved in: $OUTPUT_DIR"
echo "🔍 Summary: ${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt"
echo ""
echo "📊 To view detailed results:"
echo "  ncu-ui ${OUTPUT_DIR}/ncu_*_${TIMESTAMP}.ncu-rep"
echo ""

# Run basic Python comparison as well
echo "🐍 Running Python-based comparison for quick results..."
run_basic_comparison

echo ""
echo "🎉 All analyses completed successfully!"
echo ""
echo "Key files to examine:"
echo "  1. NCU reports: ${OUTPUT_DIR}/ncu_*_${TIMESTAMP}.ncu-rep (use ncu-ui)"
echo "  2. Python comparison: ${OUTPUT_DIR}/cuda_kernel_comparison_*.json"
echo "  3. Summary report: ${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt"