#!/bin/bash

# ST-BIF SNN NVTX Profiling Script
# Generates nsys profile and saves timing data

echo "ST-BIF SNN NVTX Profiling"
echo "========================="

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="snn_profile_${TIMESTAMP}"

echo "Output file: ${OUTPUT_FILE}.nsys-rep"
echo "Batch size: 32"
echo "Number of runs: 5"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys not found. Please ensure NVIDIA Nsight Systems is installed."
    echo "Running without nsys profiling..."
    python nsys_snn_profiling.py
    exit 0
fi

# Run nsys profiling
echo "Starting nsys profiling..."
echo "This will generate:"
echo "  1. ${OUTPUT_FILE}.nsys-rep - for GUI analysis"
echo "  2. nsys_profiling_results_*.json - detailed timing data"
echo "  3. nsys_profiling_summary_*.txt - human-readable summary"
echo ""

nsys profile \
  -o ../outputs/nsys_results/${OUTPUT_FILE} \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cusparse,cudnn,openacc,openmp,mpi,ucx \
  --sample=system-wide \
  --event-sample=system-wide \
  --cpu-core-events=0,1,2,3,4,5 \
  --gpu-metrics-device=all \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --stats=true \
  python nsys_snn_profiling.py

# nsys profile -o ../outputs/nsys_results/${OUTPUT_FILE} \
#   --trace=cuda,nvtx \
#   --capture-range=nvtx --capture-range-end=stop \
#   --cuda-event-trace=false \
#   --gpu-metrics-devices=none \
#   --sample=none --event-sample=none \
#   --stats=true \
#   python nsys_snn_profiling.py



# Check if profiling was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Profiling completed successfully!"
    echo ""
    echo "Generated files:"
    ls -la ../outputs/nsys_results/${OUTPUT_FILE}.nsys-rep 2>/dev/null && echo "  - ${OUTPUT_FILE}.nsys-rep (nsys GUI file)"
    ls -la ../outputs/nsys_results/nsys_profiling_results_*.json 2>/dev/null | tail -1 | awk '{print "  - " $9 " (detailed JSON data)"}'
    ls -la ../outputs/nsys_results/nsys_profiling_summary_*.txt 2>/dev/null | tail -1 | awk '{print "  - " $9 " (summary report)"}'
    
    echo ""
    echo "To view in Nsight Systems GUI:"
    echo "  nsight-sys ${OUTPUT_FILE}.nsys-rep"
    echo ""
    echo "Key NVTX ranges to examine:"
    echo "  - Complete_Profiling_Session (top level)"
    echo "  - SNN_Inference_Run_X (individual runs)"
    echo "  - Model_Forward_Pass (main computation)"
    echo "  - Time_Encoding (input processing)" 
    echo "  - LayerX_ResBlocks (layer breakdown)"
    echo ""
    
    # Show summary if available
    SUMMARY_FILE=$(ls -t ../outputs/nsys_results/nsys_profiling_summary_*.txt 2>/dev/null | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        echo "Quick Summary:"
        echo "=============="
        grep -A 10 "Overall Performance:" "$SUMMARY_FILE"
    fi
else
    echo "❌ Profiling failed!"
    exit 1
fi