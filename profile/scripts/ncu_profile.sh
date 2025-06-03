#!/bin/bash
# NVIDIA Nsight Compute Profiling Script for SNN
# This script runs detailed kernel-level profiling with ncu

set -e

# Default parameters
BATCH_SIZE=16  # Smaller batch for detailed kernel analysis
STEPS=20       # Fewer steps for detailed analysis
WARMUP=10
OUTPUT_DIR="profile/outputs"
SCRIPT_DIR="profile/scripts"
KERNEL_REGEX=".*"  # Profile all kernels by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --kernel-regex)
            KERNEL_REGEX="$2"
            shift 2
            ;;
        --snn-kernels)
            # Focus on SNN-specific kernels
            KERNEL_REGEX=".*st_bif.*|.*spiking.*|.*cupy.*"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --batch-size NUM     Batch size for profiling (default: 16)"
            echo "  --steps NUM          Number of profiling steps (default: 20)"
            echo "  --warmup NUM         Warmup steps (default: 10)"
            echo "  --output-dir DIR     Output directory (default: profile/outputs)"
            echo "  --kernel-regex RE    Regex to filter kernels (default: all)"
            echo "  --snn-kernels        Focus on SNN-specific kernels"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ Error: ncu (NVIDIA Nsight Compute) not found"
    echo "Please install NVIDIA Nsight Compute or ensure it's in your PATH"
    exit 1
fi

# Check if CUDA is available
if ! nvidia-smi &> /dev/null; then
    echo "❌ Error: NVIDIA GPU not available"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for unique filenames
TIMESTAMP=$(date +%s)
NCU_OUTPUT="$OUTPUT_DIR/ncu_profile_${TIMESTAMP}"

echo "🚀 Starting NVIDIA Nsight Compute Profiling"
echo "================================================"
echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Steps: $STEPS"
echo "  Warmup: $WARMUP"
echo "  Kernel Filter: $KERNEL_REGEX"
echo "  Output: $NCU_OUTPUT.ncu-rep"
echo ""
echo "⚠️  Note: NCU profiling may be slow due to detailed kernel analysis"
echo ""

# Run ncu profiling with comprehensive metrics
echo "📊 Launching ncu profiler..."
ncu \
    --set full \
    --kernel-regex="$KERNEL_REGEX" \
    --launch-skip-before-match 0 \
    --launch-count 10 \
    --target-processes application-only \
    --force-overwrite \
    --export="$NCU_OUTPUT" \
    python "$SCRIPT_DIR/snn_profiler.py" \
        --method benchmark \
        --batch-size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --device cuda

echo ""
echo "✅ Nsight Compute profiling completed!"
echo "📁 Results saved to: $NCU_OUTPUT.ncu-rep"
echo ""

# Generate summary report
echo "📈 Generating summary report..."
SUMMARY_FILE="$OUTPUT_DIR/ncu_summary_${TIMESTAMP}.txt"

{
    echo "=== NVIDIA Nsight Compute Summary ==="
    echo "Profiling Date: $(date)"
    echo "Configuration: Batch Size=$BATCH_SIZE, Steps=$STEPS, Warmup=$WARMUP"
    echo "Kernel Filter: $KERNEL_REGEX"
    echo ""
    
    # Extract basic metrics using ncu command line
    echo "Top kernels by execution time:"
    ncu --csv --page details --details-all "$NCU_OUTPUT.ncu-rep" 2>/dev/null | head -20 || echo "Detailed CSV export not available"
    echo ""
    
    echo "Full detailed report available in: $NCU_OUTPUT.ncu-rep"
    echo "Open with: ncu-ui $NCU_OUTPUT.ncu-rep"
    
} > "$SUMMARY_FILE"

echo "📄 Summary saved to: $SUMMARY_FILE"

echo ""
echo "📖 To view results:"
echo "  ncu-ui $NCU_OUTPUT.ncu-rep"
echo "  Or open with NVIDIA Nsight Compute GUI"
echo ""
echo "🎯 Analysis recommendations:"
echo "  1. Look for compute throughput bottlenecks"
echo "  2. Check memory bandwidth utilization"
echo "  3. Identify warp execution efficiency"
echo "  4. Focus on ST-BIF neuron kernel performance"
echo "  5. Compare forward vs backward pass kernels"

# Additional targeted profiling suggestions
echo ""
echo "🔍 For targeted analysis, try:"
echo "  $0 --snn-kernels --steps 10    # Focus on SNN kernels"
echo "  $0 --kernel-regex '.*conv.*'   # Focus on convolution kernels"
echo "  $0 --kernel-regex '.*gemm.*'   # Focus on matrix multiplication"