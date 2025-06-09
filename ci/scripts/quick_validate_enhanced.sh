#!/bin/bash
# ST-BIF Enhanced Quick Validation Script
# 使用增强报告系统的快速验证脚本

set -e

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_ROOT="${PROJECT_ROOT}/ci"
RESULTS_DIR="${CI_ROOT}/results/latest"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

log_info "启动ST-BIF增强快速验证"
cd "${PROJECT_ROOT}"

# 1. 环境快速检查
log_info "=== 环境快速检查 ==="

# 检查Python和CUDA
python -c "
import torch
print(f'Python版本: {torch.__version__.split(\"+\")[0]}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数: {torch.cuda.device_count()}')
    print(f'当前CUDA设备: {torch.cuda.get_device_name()}')
"

# 2. 使用pytest快速模式运行测试
log_info "=== 运行pytest快速测试套件 ==="

# 运行增强的pytest CI
python ci/scripts/run_pytest_ci.py --mode quick --output "${RESULTS_DIR}" --report-format enhanced

# 检查退出状态
EXIT_CODE=$?

echo
log_info "=== 验证完成 ==="

if [ $EXIT_CODE -eq 0 ]; then
    log_success "✅ 快速验证通过！"
    log_info "📊 详细报告已生成:"
    log_info "   - HTML: ${RESULTS_DIR}/enhanced_report.html"
    log_info "   - Markdown: ${RESULTS_DIR}/enhanced_report.md"
    log_info "   - JSON: ${RESULTS_DIR}/pytest_ci_results.json"
else
    log_error "❌ 快速验证失败！"
    log_info "📊 失败详情报告已生成:"
    log_info "   - HTML: ${RESULTS_DIR}/enhanced_report.html"
    log_info "   - Markdown: ${RESULTS_DIR}/enhanced_report.md"
    log_info "   - JSON: ${RESULTS_DIR}/pytest_ci_results.json"
fi

exit $EXIT_CODE