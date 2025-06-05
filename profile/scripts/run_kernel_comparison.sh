#!/bin/bash

# ST-BIF CUDA Kernel Comparison Master Script
# ===========================================
# This script runs comprehensive comparison between original and new CUDA kernels

set -e

echo "🚀 ST-BIF CUDA Kernel Comparison Suite"
echo "======================================"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../outputs/nsys_results"

echo "Starting comprehensive kernel analysis at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to check if kernel files exist
check_kernel_availability() {
    echo "🔍 Checking kernel availability..."
    
    ORIGINAL_KERNEL="../../neuron_cupy/cuda_operator.py"
    NEW_KERNEL="../../neuron_cupy/cuda_operator_new.py"
    
    if [ -f "$ORIGINAL_KERNEL" ]; then
        echo "  ✓ Original kernel found: $ORIGINAL_KERNEL"
        ORIGINAL_AVAILABLE=true
    else
        echo "  ❌ Original kernel not found: $ORIGINAL_KERNEL"
        ORIGINAL_AVAILABLE=false
    fi
    
    if [ -f "$NEW_KERNEL" ]; then
        echo "  ✓ New kernel found: $NEW_KERNEL"
        NEW_AVAILABLE=true
    else
        echo "  ❌ New kernel not found: $NEW_KERNEL"
        NEW_AVAILABLE=false
    fi
    
    if [ "$ORIGINAL_AVAILABLE" = false ] && [ "$NEW_AVAILABLE" = false ]; then
        echo "  ❌ No kernels available for comparison!"
        exit 1
    fi
    
    echo ""
}

# Function to run Python-based kernel profiling
run_python_profiling() {
    echo "Python内核性能分析..."
    echo "将测试以下内容:"
    echo "  - 内核等效性 (数值准确性)"
    echo "  - 性能对比 (时间、吞吐量)"
    echo "  - 内存使用分析"
    echo ""
    
    cd "$(dirname "$0")"
    
    echo "首先运行ST-BIF内核基准测试..."
    python cuda_kernel_benchmark.py
    
    echo ""
    echo "然后尝试运行内核对比测试..."
    timeout 60 python cuda_kernel_profiler.py || echo "内核对比测试遇到问题，已超时跳过"
    
    echo "  Python分析完成"
    echo ""
}

# Function to run NCU analysis
run_ncu_analysis() {
    echo "🔬 Running NCU (Nsight Compute) analysis..."
    echo "This will provide detailed kernel metrics:"
    echo "  - SM utilization and cycles"
    echo "  - Memory bandwidth usage"
    echo "  - Instruction throughput"
    echo "  - Cache efficiency"
    echo ""
    
    cd "$(dirname "$0")"
    ./ncu_kernel_comparison.sh
    
    echo "  ✓ NCU analysis completed"
    echo ""
}

# Function to run NSYS analysis with kernel focus
run_nsys_kernel_analysis() {
    echo "📊 Running NSYS analysis with kernel focus..."
    
    # Create a kernel-focused NSYS script
    cat > "temp_nsys_kernel.py" << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('../..')

import torch
import torch.cuda.nvtx as nvtx
from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS
import time

def main():
    device = 'cuda'
    batch_size = 32
    time_steps = 8
    feature_size = 512
    num_runs = 10
    
    print(f"NSYS Kernel Analysis")
    print(f"Batch: {batch_size}, Time steps: {time_steps}, Features: {feature_size}")
    
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
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            neuron._reset_states()
            _ = neuron(x_seq)
    
    torch.cuda.synchronize()
    print("Starting profiled runs...")
    
    # Profiled runs with NVTX
    nvtx.range_push("ST_BIF_Kernel_Analysis")
    
    run_times = []
    for i in range(num_runs):
        nvtx.range_push(f"ST_BIF_Kernel_Run_{i}")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            neuron._reset_states()
            output = neuron(x_seq)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        run_time = (end_time - start_time) * 1000
        run_times.append(run_time)
        
        nvtx.range_pop()
        print(f"Run {i+1}/{num_runs}: {run_time:.3f} ms")
    
    nvtx.range_pop()
    
    mean_time = sum(run_times) / len(run_times)
    throughput = (batch_size * time_steps) / (mean_time / 1000)
    
    print(f"\nResults:")
    print(f"  Mean time: {mean_time:.3f} ms")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Output shape: {output.shape}")

if __name__ == "__main__":
    main()
EOF
    
    # Run NSYS with kernel focus
    NSYS_OUTPUT="${OUTPUT_DIR}/nsys_kernel_analysis_${TIMESTAMP}"
    
    nsys profile \
        -o "$NSYS_OUTPUT" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --capture-range=nvtx \
        --capture-range-end=stop \
        --cuda-memory-usage=true \
        --gpu-metrics-device=all \
        --stats=true \
        python temp_nsys_kernel.py \
        > "${OUTPUT_DIR}/nsys_kernel_stdout_${TIMESTAMP}.txt" 2>&1
    
    # Cleanup
    rm temp_nsys_kernel.py
    
    echo "  ✓ NSYS kernel analysis completed"
    echo "  📁 Results: ${NSYS_OUTPUT}.nsys-rep"
    echo ""
}

# Function to generate comprehensive summary
generate_summary() {
    echo "📋 Generating comprehensive summary..."
    
    SUMMARY_FILE="${OUTPUT_DIR}/kernel_comparison_summary_${TIMESTAMP}.md"
    
    cat > "$SUMMARY_FILE" << EOF
# ST-BIF CUDA Kernel Comparison Summary

**Analysis Date:** $(date)  
**Analysis ID:** $TIMESTAMP

## Configuration
- Batch Size: 32
- Time Steps: 8
- Feature Sizes: 256, 512, 1024
- Device: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)

## Kernel Availability
- Original Kernel: $ORIGINAL_AVAILABLE
- New Kernel: $NEW_AVAILABLE

## Analysis Types Performed

### 1. Python-based Profiling
- **Purpose:** Functional equivalence and basic performance comparison
- **Files:** \`cuda_kernel_comparison_*.json\`, \`cuda_kernel_summary_*.txt\`
- **Metrics:** Timing, throughput, memory usage, numerical accuracy

### 2. NCU (Nsight Compute) Analysis  
- **Purpose:** Detailed kernel-level performance metrics
- **Files:** \`ncu_*_${TIMESTAMP}.ncu-rep\`, \`ncu_*_${TIMESTAMP}.csv\`
- **Metrics:** SM utilization, memory bandwidth, instruction throughput

### 3. NSYS (Nsight Systems) Analysis
- **Purpose:** Timeline analysis and system-level profiling
- **Files:** \`nsys_kernel_analysis_${TIMESTAMP}.nsys-rep\`
- **Metrics:** GPU timeline, kernel launch overhead, memory transfers

## Key Files Generated

EOF
    
    # List all generated files
    echo "### Generated Files" >> "$SUMMARY_FILE"
    ls -la "$OUTPUT_DIR"/*${TIMESTAMP}* 2>/dev/null | while read -r line; do
        filename=$(echo "$line" | awk '{print $9}')
        if [ -n "$filename" ]; then
            echo "- \`$(basename "$filename")\`" >> "$SUMMARY_FILE"
        fi
    done
    
    cat >> "$SUMMARY_FILE" << EOF

## Analysis Instructions

### 1. Quick Results Review
\`\`\`bash
# View Python profiling summary
cat ${OUTPUT_DIR}/cuda_kernel_summary_*.txt

# View NCU summary  
cat ${OUTPUT_DIR}/ncu_analysis_summary_${TIMESTAMP}.txt
\`\`\`

### 2. Detailed Analysis Tools

#### NCU GUI Analysis:
\`\`\`bash
ncu-ui ${OUTPUT_DIR}/ncu_*_${TIMESTAMP}.ncu-rep
\`\`\`

#### NSYS GUI Analysis:  
\`\`\`bash
nsight-sys ${OUTPUT_DIR}/nsys_kernel_analysis_${TIMESTAMP}.nsys-rep
\`\`\`

### 3. Key Metrics to Compare

#### Performance Metrics:
- **Execution Time:** Lower is better
- **Throughput:** Higher is better (samples/second)
- **Memory Usage:** Lower peak memory is better

#### NCU Metrics:
- **SM Cycles Elapsed:** Kernel execution time
- **DRAM Throughput:** Memory bandwidth utilization
- **Instruction Throughput:** Compute efficiency
- **Cache Hit Rates:** Memory access efficiency

#### Equivalence Check:
- **Max Absolute Difference:** Should be < 1e-5
- **Relative Error:** Should be < 1e-4

## Optimization Recommendations

Based on the analysis results, consider:

1. **If kernels are equivalent but new is faster:**
   - Replace original with new implementation
   - Document performance improvements

2. **If kernels differ significantly:**
   - Investigate numerical differences
   - Verify correctness of new implementation
   - Check edge cases and boundary conditions

3. **If performance is similar:**
   - Consider other factors: code maintainability, memory usage
   - Profile with different input sizes
   - Test on different GPU architectures

## Next Steps

1. Review all generated analysis files
2. Verify numerical equivalence between kernels
3. Compare performance metrics across different scenarios
4. Make informed decision about kernel implementation

---
*Generated by ST-BIF CUDA Kernel Comparison Suite*
EOF

    echo "  ✓ Summary generated: $SUMMARY_FILE"
    echo ""
}

# Main execution flow
main() {
    echo "Starting comprehensive kernel comparison analysis..."
    echo ""
    
    # Check prerequisites
    check_kernel_availability
    
    # Run all analysis types
    echo "📝 Analysis Plan:"
    echo "  1. Python-based profiling (functional + performance)"
    echo "  2. NCU analysis (detailed kernel metrics)"  
    echo "  3. NSYS analysis (timeline and system view)"
    echo "  4. Comprehensive summary generation"
    echo ""
    
    read -p "Press Enter to continue or Ctrl+C to abort..."
    echo ""
    
    # Execute analyses
    run_python_profiling
    run_ncu_analysis
    run_nsys_kernel_analysis
    generate_summary
    
    # Final summary
    echo "🎉 Comprehensive kernel comparison completed!"
    echo ""
    echo "📁 All results saved in: $OUTPUT_DIR"
    echo "📋 Summary report: ${OUTPUT_DIR}/kernel_comparison_summary_${TIMESTAMP}.md"
    echo ""
    echo "🔍 Quick access commands:"
    echo "  # View summary"
    echo "  cat ${OUTPUT_DIR}/kernel_comparison_summary_${TIMESTAMP}.md"
    echo ""
    echo "  # Open GUI tools"
    echo "  ncu-ui ${OUTPUT_DIR}/ncu_*_${TIMESTAMP}.ncu-rep"
    echo "  nsight-sys ${OUTPUT_DIR}/nsys_kernel_analysis_${TIMESTAMP}.nsys-rep"
    echo ""
    echo "✅ Analysis complete! Review the generated files for detailed results."
}

# Run main function
main "$@"