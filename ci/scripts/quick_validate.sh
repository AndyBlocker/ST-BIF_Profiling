#!/bin/bash
# ST-BIF Quick Validation Script
# 快速验证脚本，适合日常开发使用 (1-2分钟完成)

set -e  # 遇到错误立即退出

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_ROOT="${PROJECT_ROOT}/ci"
RESULTS_DIR="${CI_ROOT}/results/latest"
CONFIG_FILE="${CI_ROOT}/configs/validation_config.yaml"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

log_info "启动ST-BIF快速验证 (L0级别)"
log_info "项目根目录: ${PROJECT_ROOT}"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 初始化结果追踪
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# 函数：运行检查并记录结果
run_check() {
    local check_name="$1"
    local command="$2"
    local timeout="${3:-60}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "运行检查: ${check_name}"
    
    if timeout "${timeout}" bash -c "${command}" &> "${RESULTS_DIR}/${check_name}.log"; then
        log_success "${check_name}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        log_error "${check_name} - 查看日志: ${RESULTS_DIR}/${check_name}.log"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# 1. 环境检查
log_info "=== 环境检查 ==="

# 检查Python环境
run_check "python_env" "python --version && python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"

# 检查CUDA可用性
run_check "cuda_check" "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA devices: {torch.cuda.device_count()}\")'"

# 2. 导入测试
log_info "=== 导入测试 ==="

# 核心模块导入测试
run_check "import_snn" "python -c 'from snn import ST_BIFNeuron_MS, MyQuan; print(\"SNN模块导入成功\")'"

run_check "import_models" "python -c 'from models.resnet import resnet18; print(\"Models模块导入成功\")'"

run_check "import_wrapper" "python -c 'from wrapper import SNNWrapper_MS; print(\"Wrapper模块导入成功\")'"

run_check "import_utils" "python -c 'from utils import functions, io, misc; print(\"Utils模块导入成功\")'"

# CUDA算子导入测试 (允许失败，如果无CUDA)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    run_check "import_cuda_operator" "python -c 'from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA; print(\"CUDA算子导入成功\")'"
else
    log_warning "跳过CUDA算子导入测试 (CUDA不可用)"
fi

# 3. 语法和代码风格检查
log_info "=== 代码质量检查 ==="

# Python语法检查
run_check "syntax_check" "python -m py_compile snn/__init__.py models/__init__.py wrapper/__init__.py utils/__init__.py"

# 4. 基础功能测试
log_info "=== 基础功能测试 ==="

# 快速模型加载测试
if [ -f "checkpoints/resnet/best_ANN.pth" ]; then
    run_check "model_loading" "python -c '
import torch
from models.resnet import resnet18
model = resnet18(num_classes=10)
checkpoint = torch.load(\"checkpoints/resnet/best_ANN.pth\", map_location=\"cpu\")
model.load_state_dict(checkpoint)
print(\"模型加载成功\")
'"
else
    log_warning "跳过模型加载测试 (检查点文件不存在)"
fi

# SNN包装器功能测试 (简化版本)
run_check "snn_wrapper_basic" "python -c '
import torch
from wrapper import SNNWrapper_MS
from models.resnet import resnet18

# 创建简单模型
model = resnet18(num_classes=10)
# 测试SNN包装器初始化
snn_wrapper = SNNWrapper_MS(
    ann_model=model, 
    cfg=None, 
    time_step=4,
    Encoding_type=\"analog\", 
    level=8
)
print(\"SNN包装器初始化成功\")
'"

# 5. 快速CUDA算子测试 (如果可用)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    log_info "=== CUDA算子快速测试 ==="
    
    run_check "cuda_operator_import" "python -c '
import torch
import sys
sys.path.append(\"neuron_cupy\")

if torch.cuda.is_available():
    try:
        from cuda_operator import ST_BIFNodeATGF_MS_CUDA
        print(\"CUDA算子类导入成功\")
        # 验证CUDA编译
        if hasattr(ST_BIFNodeATGF_MS_CUDA, \"cuda_source\") and ST_BIFNodeATGF_MS_CUDA.cuda_source:
            print(\"CUDA内核已编译\")
        else:
            print(\"CUDA内核未编译，但类导入成功\")
    except Exception as e:
        print(f\"CUDA算子导入失败: {e}\")
        exit(1)
else:
    print(\"CUDA不可用，跳过CUDA算子测试\")
'" 60
else
    log_warning "跳过CUDA算子测试 (CUDA不可用)"
fi

# 6. 结果汇总
log_info "=== 验证结果汇总 ==="

# 生成结果报告
cat > "${RESULTS_DIR}/quick_validation_summary.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "validation_level": "L0_quick",
    "total_checks": ${TOTAL_CHECKS},
    "passed_checks": ${PASSED_CHECKS},
    "failed_checks": ${FAILED_CHECKS},
    "success_rate": $(echo "scale=2; ${PASSED_CHECKS} * 100 / ${TOTAL_CHECKS}" | bc -l),
    "duration_seconds": ${SECONDS},
    "status": "$([ ${FAILED_CHECKS} -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF

# 显示结果
echo
echo "================================="
if [ ${FAILED_CHECKS} -eq 0 ]; then
    log_success "快速验证通过！"
    log_success "检查通过: ${PASSED_CHECKS}/${TOTAL_CHECKS}"
    log_success "耗时: ${SECONDS}秒"
    exit 0
else
    log_error "快速验证失败！"
    log_error "检查通过: ${PASSED_CHECKS}/${TOTAL_CHECKS}"
    log_error "检查失败: ${FAILED_CHECKS}/${TOTAL_CHECKS}"
    log_error "耗时: ${SECONDS}秒"
    echo
    log_info "失败详情请查看: ${RESULTS_DIR}/*.log"
    exit 1
fi