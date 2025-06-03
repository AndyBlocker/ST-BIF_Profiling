#!/bin/bash
# NVIDIA Nsight Systems Profiling Script for SNN
# This script runs the SNN profiler with nsys to capture GPU activity

set -e

# Default parameters
BATCH_SIZE=32
STEPS=100
WARMUP=20
OUTPUT_DIR="profile/outputs"
SCRIPT_DIR="profile/scripts"

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --batch-size NUM    Batch size for profiling (default: 32)"
            echo "  --steps NUM         Number of profiling steps (default: 100)"
            echo "  --warmup NUM        Warmup steps (default: 20)"
            echo "  --output-dir DIR    Output directory (default: profile/outputs)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "❌ Error: nsys (NVIDIA Nsight Systems) not found"
    echo "Please install NVIDIA Nsight Systems or ensure it's in your PATH"
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
NSYS_OUTPUT="$OUTPUT_DIR/nsys_profile_${TIMESTAMP}"

echo "🚀 Starting NVIDIA Nsight Systems Profiling"
echo "================================================"
echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Steps: $STEPS"
echo "  Warmup: $WARMUP"
echo "  Output: $NSYS_OUTPUT.nsys-rep"
echo ""

# Run nsys profiling
echo "📊 Launching nsys profiler..."
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --export=sqlite \
    --output="$NSYS_OUTPUT" \
    python "$SCRIPT_DIR/snn_profiler.py" \
        --method benchmark \
        --batch-size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --device cuda

echo ""
echo "✅ Nsight Systems profiling completed!"
echo "📁 Results saved to:"
echo "  - $NSYS_OUTPUT.nsys-rep (main report)"
echo "  - $NSYS_OUTPUT.sqlite (database export)"
echo ""
echo "📖 To view results:"
echo "  nsys-ui $NSYS_OUTPUT.nsys-rep"
echo "  Or open with NVIDIA Nsight Systems GUI"

# Generate comprehensive text reports
if command -v nsys &> /dev/null; then
    echo ""
    echo "📈 Generating detailed text reports..."
    
    # Main summary file
    SUMMARY_FILE="$OUTPUT_DIR/nsys_summary_${TIMESTAMP}.txt"
    
    # Detailed analysis files
    GPU_TRACE_FILE="$OUTPUT_DIR/nsys_gputrace_${TIMESTAMP}.txt"
    CUDA_API_FILE="$OUTPUT_DIR/nsys_cudaapi_${TIMESTAMP}.txt"
    MEMORY_FILE="$OUTPUT_DIR/nsys_memory_${TIMESTAMP}.txt"
    KERNEL_FILE="$OUTPUT_DIR/nsys_kernels_${TIMESTAMP}.txt"
    
    echo "📊 Extracting GPU trace data..."
    {
        echo "=== GPU Trace Statistics ==="
        echo "Profiling Date: $(date)"
        echo "Configuration: Batch Size=$BATCH_SIZE, Steps=$STEPS, Warmup=$WARMUP"
        echo ""
        nsys stats --report gputrace --format table "$NSYS_OUTPUT.nsys-rep"
        echo ""
        echo "=== GPU Trace CSV Data ==="
        nsys stats --report gputrace --format csv "$NSYS_OUTPUT.nsys-rep"
    } > "$GPU_TRACE_FILE"
    
    echo "📊 Extracting CUDA API data..."
    {
        echo "=== CUDA API Summary ==="
        echo "Profiling Date: $(date)"
        echo ""
        nsys stats --report cudaapisum --format table "$NSYS_OUTPUT.nsys-rep"
        echo ""
        echo "=== CUDA API CSV Data ==="
        nsys stats --report cudaapisum --format csv "$NSYS_OUTPUT.nsys-rep"
    } > "$CUDA_API_FILE"
    
    echo "📊 Extracting memory operations..."
    {
        echo "=== GPU Memory Operations ==="
        echo "Profiling Date: $(date)"
        echo ""
        nsys stats --report gpumemtimesum --format table "$NSYS_OUTPUT.nsys-rep" 2>/dev/null || echo "Memory time summary not available"
        echo ""
        nsys stats --report gpumemsizesum --format table "$NSYS_OUTPUT.nsys-rep" 2>/dev/null || echo "Memory size summary not available"
    } > "$MEMORY_FILE"
    
    echo "📊 Extracting kernel details..."
    {
        echo "=== GPU Kernel Details ==="
        echo "Profiling Date: $(date)"
        echo ""
        echo "Top GPU kernels by duration:"
        nsys stats --report gputrace --format csv "$NSYS_OUTPUT.nsys-rep" | \
            grep -v "^Start" | head -50 | \
            awk -F',' 'BEGIN{OFS=","} {if(NR==1) print "Duration(ns),Kernel_Name"; else if($7!="") printf "%.0f,%s\n", $7, $13}' | \
            sort -t',' -k1 -nr | head -20
        echo ""
        echo "SNN-specific kernels (ST-BIF related):"
        nsys stats --report gputrace --format csv "$NSYS_OUTPUT.nsys-rep" | \
            grep -i -E "(bif|spike|neuron|snn)" || echo "No SNN-specific kernels found"
    } > "$KERNEL_FILE"
    
    echo "📊 Creating comprehensive summary..."
    {
        echo "=== NVIDIA Nsight Systems Analysis Summary ==="
        echo "Profiling Date: $(date)"
        echo "Configuration: Batch Size=$BATCH_SIZE, Steps=$STEPS, Warmup=$WARMUP"
        echo "Output Files: $NSYS_OUTPUT.*"
        echo ""
        
        echo "=== Top 10 GPU Kernels by Duration ==="
        nsys stats --report gputrace --format csv "$NSYS_OUTPUT.nsys-rep" | \
            awk -F',' 'NR>1 && $7!="" {printf "%.2f ms - %s\n", $7/1000000, $13}' | \
            sort -nr | head -10
        echo ""
        
        echo "=== CUDA API Call Summary ==="
        nsys stats --report cudaapisum --format csv "$NSYS_OUTPUT.nsys-rep" | \
            awk -F',' 'NR>1 && $4!="" {printf "%s: %.2f ms (avg: %.2f ms)\n", $1, $4/1000000, ($4/$3)/1000000}' | \
            head -15
        echo ""
        
        echo "=== Performance Metrics ==="
        echo "Total GPU kernels: $(nsys stats --report gputrace --format csv "$NSYS_OUTPUT.nsys-rep" | wc -l)"
        echo "Analysis files generated:"
        echo "  - GPU Trace: $GPU_TRACE_FILE"
        echo "  - CUDA API: $CUDA_API_FILE"
        echo "  - Memory Ops: $MEMORY_FILE" 
        echo "  - Kernel Details: $KERNEL_FILE"
        echo ""
        echo "Full report: $NSYS_OUTPUT.nsys-rep"
        
    } > "$SUMMARY_FILE"
    
    echo "📄 Text analysis files generated:"
    echo "  - Summary: $SUMMARY_FILE"
    echo "  - GPU Trace: $GPU_TRACE_FILE"
    echo "  - CUDA API: $CUDA_API_FILE"
    echo "  - Memory Ops: $MEMORY_FILE"
    echo "  - Kernel Details: $KERNEL_FILE"
fi

echo ""
echo "🎯 Next steps:"
echo "  1. Open $NSYS_OUTPUT.nsys-rep in Nsight Systems GUI"
echo "  2. Focus on CUDA kernels and memory transfers"
echo "  3. Look for bottlenecks in ST-BIF neuron operations"