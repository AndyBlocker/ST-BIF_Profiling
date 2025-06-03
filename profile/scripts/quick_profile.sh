#!/bin/bash
# Quick profiling script for development workflow
# This script provides fast profiling for iterative development

set -e

echo "🚀 ST-BIF SNN Quick Profiling"
echo "=============================="

# Quick benchmark
echo "⚡ Running quick benchmark..."
python profile/scripts/snn_profiler.py \
    --method benchmark \
    --batch-size 16 \
    --steps 20 \
    --warmup 5

echo ""
echo "✅ Quick profiling completed!"
echo "📁 Results saved in: profile/outputs/"
echo ""
echo "🔄 For comprehensive profiling, run:"
echo "  ./profile/scripts/run_all_profiles.sh"