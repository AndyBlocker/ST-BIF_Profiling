#!/bin/bash
# ANN vs SNN Performance Comparison Script
# This script runs comprehensive comparison between ANN and SNN models
# for both offline (batch=32) and online (batch=1) scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

# Default parameters
METHODS="benchmark"
DEVICE="cuda"
OUTPUT_DIR="profile/outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            # Quick mode - benchmark only
            METHODS="benchmark"
            shift 1
            ;;
        --comprehensive)
            # Comprehensive mode - all methods
            METHODS="torch memory benchmark"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --methods METHOD    Profiling methods (benchmark/torch/memory/all)"
            echo "  --device DEVICE     Device to use (cuda/cpu)"
            echo "  --output-dir DIR    Output directory"
            echo "  --quick             Quick benchmark comparison only"
            echo "  --comprehensive     Comprehensive profiling (all methods)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "This script compares ANN vs SNN performance in two scenarios:"
            echo "  1. Offline (batch_size=32) - Batch processing"
            echo "  2. Online (batch_size=1)   - Real-time streaming"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "ANN vs SNN Performance Comparison"
echo "Configuration:"
echo "  Methods: $METHODS"
echo "  Device: $DEVICE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""
echo "Scenarios:"
echo "  📦 Offline Processing (batch_size=32)"
echo "  🌊 Online Streaming (batch_size=1)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python environment
if ! python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
    print_error "PyTorch not available"
    exit 1
fi

# Check CUDA if specified
if [ "$DEVICE" = "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_warning "CUDA not available, switching to CPU"
        DEVICE="cpu"
    fi
fi

echo ""

# Phase 1: ANN Profiling
print_header "Phase 1: Standard ANN Profiling"
echo "Running baseline ANN performance analysis..."

print_info "🔄 Running ANN profiler for both scenarios..."
python profile/scripts/ann_profiler.py \
    --scenario both \
    --methods $METHODS \
    --device "$DEVICE"

print_success "ANN profiling completed"
echo ""

# Phase 2: SNN Profiling
print_header "Phase 2: SNN Profiling"
echo "Running SNN performance analysis..."

# Offline scenario (batch_size=32)
print_info "📦 Running SNN offline scenario (batch_size=32)..."
python profile/scripts/snn_profiler.py \
    --method $METHODS \
    --batch-size 32 \
    --steps 100 \
    --warmup 20 \
    --device "$DEVICE"

print_success "SNN offline scenario completed"

# Online scenario (batch_size=1)  
print_info "🌊 Running SNN online scenario (batch_size=1)..."
python profile/scripts/snn_profiler.py \
    --method $METHODS \
    --batch-size 1 \
    --steps 200 \
    --warmup 20 \
    --device "$DEVICE"

print_success "SNN online scenario completed"
echo ""

# Phase 3: Comparative Analysis
print_header "Phase 3: Comparative Analysis"
echo "Generating comprehensive comparison report..."

# Generate comparison report
TIMESTAMP=$(date +%s)
COMPARISON_REPORT="$OUTPUT_DIR/ann_vs_snn_comparison_${TIMESTAMP}.txt"

{
    echo "ANN vs SNN PERFORMANCE COMPARISON REPORT"
    echo "========================================"
    echo "Generated: $(date)"
    echo "Device: $DEVICE"
    echo "Methods: $METHODS"
    echo ""
    
    echo "SCENARIO OVERVIEW:"
    echo "  📦 Offline Processing: batch_size=32, optimized for throughput"
    echo "  🌊 Online Streaming:   batch_size=1,  optimized for latency"
    echo ""
    
    echo "BENCHMARK RESULTS:"
    echo "=================="
    
    # Extract latest benchmark results
    echo ""
    echo "ANN Performance:"
    echo "----------------"
    
    # Find latest ANN results
    ANN_OFFLINE=$(ls -t "$OUTPUT_DIR"/ann_benchmark_offline_bs32_*.txt 2>/dev/null | head -1)
    ANN_ONLINE=$(ls -t "$OUTPUT_DIR"/ann_benchmark_online_bs1_*.txt 2>/dev/null | head -1)
    
    if [ -n "$ANN_OFFLINE" ]; then
        echo "📦 Offline (batch=32):"
        grep -A 10 "Timing Results" "$ANN_OFFLINE" | head -15
        echo ""
    fi
    
    if [ -n "$ANN_ONLINE" ]; then
        echo "🌊 Online (batch=1):"
        grep -A 10 "Timing Results" "$ANN_ONLINE" | head -15
        echo ""
    fi
    
    echo "SNN Performance:"
    echo "----------------"
    
    # Find latest SNN results
    SNN_OFFLINE=$(ls -t "$OUTPUT_DIR"/benchmark_*bs32*.txt 2>/dev/null | head -1)
    SNN_ONLINE=$(ls -t "$OUTPUT_DIR"/benchmark_*bs1*.txt 2>/dev/null | head -1)
    
    if [ -z "$SNN_OFFLINE" ]; then
        # Try without bs32 pattern
        SNN_OFFLINE=$(ls -t "$OUTPUT_DIR"/benchmark_*.txt 2>/dev/null | head -1)
    fi
    
    if [ -n "$SNN_OFFLINE" ]; then
        echo "📦 Offline (batch=32) - Latest SNN result:"
        grep -A 10 "Configuration" "$SNN_OFFLINE" | head -15
        echo ""
    fi
    
    if [ -n "$SNN_ONLINE" ]; then
        echo "🌊 Online (batch=1) - Latest SNN result:"
        grep -A 10 "Configuration" "$SNN_ONLINE" | head -15
        echo ""
    fi
    
    echo "ANALYSIS & INSIGHTS:"
    echo "==================="
    echo ""
    echo "Performance Characteristics:"
    echo "  • ANN models typically show 15-25% forward, 75-85% backward time"
    echo "  • SNN models have more complex temporal processing overhead"
    echo "  • Batch processing (batch=32) improves GPU utilization"
    echo "  • Single sample processing (batch=1) minimizes latency"
    echo ""
    echo "Expected Trends:"
    echo "  • ANN should show higher raw throughput"
    echo "  • SNN provides energy efficiency benefits (not measured here)"
    echo "  • Batch processing shows better samples/sec efficiency"
    echo "  • Online scenarios show true per-sample latency"
    echo ""
    echo "Use Cases:"
    echo "  📦 Offline: Training, batch inference, data analysis"
    echo "  🌊 Online:  Real-time systems, edge deployment, interactive apps"
    
} > "$COMPARISON_REPORT"

print_success "Comparison report generated: $COMPARISON_REPORT"

# Display summary
echo ""
print_header "Performance Summary"

# Quick summary from benchmark files
echo "📊 Quick Performance Overview:"
echo ""

# Function to extract throughput from benchmark file
extract_throughput() {
    local file="$1"
    if [ -f "$file" ]; then
        grep "Samples/sec:" "$file" | awk '{print $2}'
    else
        echo "N/A"
    fi
}

# Function to extract total time from benchmark file
extract_time() {
    local file="$1" 
    if [ -f "$file" ]; then
        grep "Average total time:" "$file" | awk '{print $4}'
    else
        echo "N/A"
    fi
}

# ANN Results
if [ -n "$ANN_OFFLINE" ] && [ -n "$ANN_ONLINE" ]; then
    ANN_OFF_THROUGHPUT=$(extract_throughput "$ANN_OFFLINE")
    ANN_ON_THROUGHPUT=$(extract_throughput "$ANN_ONLINE")
    ANN_OFF_TIME=$(extract_time "$ANN_OFFLINE")
    ANN_ON_TIME=$(extract_time "$ANN_ONLINE")
    
    echo "📈 ANN Performance:"
    echo "   Offline: ${ANN_OFF_THROUGHPUT} samples/sec (${ANN_OFF_TIME} ms/step)"
    echo "   Online:  ${ANN_ON_THROUGHPUT} samples/sec (${ANN_ON_TIME} ms/step)"
fi

# SNN Results (approximate from latest files)
if [ -n "$SNN_OFFLINE" ]; then
    SNN_OFF_THROUGHPUT=$(extract_throughput "$SNN_OFFLINE")
    SNN_OFF_TIME=$(extract_time "$SNN_OFFLINE")
    echo "🧠 SNN Performance (latest results):"
    echo "   Recent:  ${SNN_OFF_THROUGHPUT} samples/sec (${SNN_OFF_TIME} ms/step)"
fi

echo ""
echo "📁 Detailed Results Location:"
echo "  All files: $OUTPUT_DIR"
echo "  Comparison: $COMPARISON_REPORT"
echo ""

print_header "Analysis Complete!"
echo "🎯 Next Steps:"
echo "  1. Review comparison report: $COMPARISON_REPORT"
echo "  2. Analyze Chrome traces for detailed breakdowns"
echo "  3. Compare forward/backward time distributions"
echo "  4. Evaluate throughput vs latency trade-offs"
echo "  5. Consider energy efficiency benefits of SNN (not measured)"
echo ""
echo "📊 Key Questions to Explore:"
echo "  • How does SNN temporal processing affect latency?"
echo "  • What is the throughput cost of spike-based computation?"
echo "  • How do batch sizes affect each model type differently?"
echo "  • Where are the computational bottlenecks in each approach?"
echo ""

print_success "ANN vs SNN comparison completed successfully!"