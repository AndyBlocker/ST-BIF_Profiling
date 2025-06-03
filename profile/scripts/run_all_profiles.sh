#!/bin/bash
# Master profiling script - runs all profiling methods
# This script orchestrates comprehensive profiling of the SNN framework

set -e

# Default parameters
BATCH_SIZE=32
STEPS=100
WARMUP=20
OUTPUT_DIR="profile/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}$1${NC}"
    echo "$(printf '=%.0s' {1..60})"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

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
        --quick)
            # Quick profiling mode
            BATCH_SIZE=16
            STEPS=50
            WARMUP=10
            shift 1
            ;;
        --thorough)
            # Thorough profiling mode
            BATCH_SIZE=64
            STEPS=200
            WARMUP=50
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --batch-size NUM    Batch size for profiling (default: 32)"
            echo "  --steps NUM         Number of profiling steps (default: 100)"
            echo "  --warmup NUM        Warmup steps (default: 20)"
            echo "  --output-dir DIR    Output directory (default: profile/outputs)"
            echo "  --quick             Quick profiling mode (small batch, fewer steps)"
            echo "  --thorough          Thorough profiling mode (large batch, more steps)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "This script runs all available profiling methods:"
            echo "  1. PyTorch Profiler (torch.profiler)"
            echo "  2. NVIDIA Nsight Systems (nsys)"
            echo "  3. NVIDIA Nsight Compute (ncu)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "ST-BIF SNN Comprehensive Profiling Suite"
echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Steps: $STEPS"
echo "  Warmup: $WARMUP"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python environment
if ! python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
    print_error "PyTorch not available"
    exit 1
fi

# Check CUDA
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
    print_warning "CUDA not available, some profiling features will be limited"
fi

# Check for NVIDIA tools
NSYS_AVAILABLE=false
NCU_AVAILABLE=false

if command -v nsys &> /dev/null; then
    NSYS_AVAILABLE=true
    print_success "nsys (Nsight Systems) found"
else
    print_warning "nsys (Nsight Systems) not found"
fi

if command -v ncu &> /dev/null; then
    NCU_AVAILABLE=true
    print_success "ncu (Nsight Compute) found"
else
    print_warning "ncu (Nsight Compute) not found"
fi

echo ""

# Run profiling methods
SCRIPT_DIR="$(dirname "$0")"

# 1. PyTorch Profiler
print_header "1. PyTorch Profiler (torch.profiler)"
echo "Running comprehensive PyTorch profiling..."
python "$SCRIPT_DIR/snn_profiler.py" \
    --method all \
    --batch-size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --warmup "$WARMUP"

print_success "PyTorch profiling completed"
echo ""

# 2. NVIDIA Nsight Systems
if [ "$NSYS_AVAILABLE" = true ]; then
    print_header "2. NVIDIA Nsight Systems (nsys)"
    echo "Running GPU timeline profiling..."
    "$SCRIPT_DIR/nsys_profile.sh" \
        --batch-size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --warmup "$WARMUP" \
        --output-dir "$OUTPUT_DIR"
    
    print_success "Nsight Systems profiling completed"
else
    print_warning "Skipping Nsight Systems profiling (nsys not available)"
fi
echo ""

# 3. NVIDIA Nsight Compute
if [ "$NCU_AVAILABLE" = true ]; then
    print_header "3. NVIDIA Nsight Compute (ncu)"
    echo "Running detailed kernel profiling..."
    print_warning "This may take several minutes due to detailed analysis"
    
    # Use smaller parameters for NCU due to overhead
    NCU_BATCH_SIZE=$((BATCH_SIZE / 2))
    NCU_STEPS=$((STEPS / 5))
    
    "$SCRIPT_DIR/ncu_profile.sh" \
        --batch-size "$NCU_BATCH_SIZE" \
        --steps "$NCU_STEPS" \
        --warmup "$WARMUP" \
        --output-dir "$OUTPUT_DIR"
    
    print_success "Nsight Compute profiling completed"
else
    print_warning "Skipping Nsight Compute profiling (ncu not available)"
fi
echo ""

# Generate summary report
print_header "Summary Report"
TIMESTAMP=$(date +%s)
SUMMARY_FILE="$OUTPUT_DIR/profiling_summary_${TIMESTAMP}.txt"

{
    echo "ST-BIF SNN Profiling Summary"
    echo "============================"
    echo "Date: $(date)"
    echo "Configuration:"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Steps: $STEPS"
    echo "  Warmup: $WARMUP"
    echo ""
    
    echo "Profiling Methods Executed:"
    echo "  ✅ PyTorch Profiler (torch.profiler)"
    [ "$NSYS_AVAILABLE" = true ] && echo "  ✅ NVIDIA Nsight Systems (nsys)" || echo "  ❌ NVIDIA Nsight Systems (nsys not available)"
    [ "$NCU_AVAILABLE" = true ] && echo "  ✅ NVIDIA Nsight Compute (ncu)" || echo "  ❌ NVIDIA Nsight Compute (ncu not available)"
    echo ""
    
    echo "Output Files:"
    ls -la "$OUTPUT_DIR"/*${TIMESTAMP}* 2>/dev/null || echo "  (No timestamped files found)"
    echo ""
    
    echo "Analysis Recommendations:"
    echo "  1. Start with PyTorch profiler results for high-level bottlenecks"
    echo "  2. Use Nsight Systems for GPU utilization and memory transfer analysis"
    echo "  3. Use Nsight Compute for detailed kernel optimization"
    echo "  4. Focus on ST-BIF neuron operations and temporal processing"
    echo "  5. Compare forward vs backward pass performance"
    
} > "$SUMMARY_FILE"

echo "📄 Summary report saved to: $SUMMARY_FILE"
echo ""

print_header "Profiling Complete!"
echo "📁 All results saved in: $OUTPUT_DIR"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review PyTorch profiler traces in Chrome (chrome://tracing/)"
if [ "$NSYS_AVAILABLE" = true ]; then
    echo "  2. Open *.nsys-rep files in Nsight Systems GUI"
fi
if [ "$NCU_AVAILABLE" = true ]; then
    echo "  3. Open *.ncu-rep files in Nsight Compute GUI"
fi
echo "  4. Focus analysis on:"
echo "     - ST-BIF neuron forward/backward operations"
echo "     - Temporal encoding bottlenecks"
echo "     - CUDA kernel efficiency"
echo "     - Memory bandwidth utilization"
echo ""
echo "📊 Happy profiling!"