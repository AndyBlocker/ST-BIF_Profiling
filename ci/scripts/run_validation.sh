#!/bin/bash
# ST-BIF 分层验证脚本
# 支持指定验证层级，提供灵活的验证选项

set -e

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
PURPLE='\033[0;35m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }
log_stage() { echo -e "${PURPLE}[STAGE]${NC} $1"; }

# 使用说明
usage() {
    cat << EOF
ST-BIF 分层验证系统

用法: $0 [选项]

验证层级:
  --level L0    快速验证层 (1-2分钟)
  --level L1    算子验证层 (5-10分钟)
  --level L2    模型验证层 (10-20分钟)
  --level L3    性能验证层 (20-60分钟)
  --level L4    深度验证层 (小时级)
  --all         运行所有层级 (默认到L3)

选项:
  --quick       仅运行L0和L1层级
  --no-cuda     跳过CUDA相关测试
  --batch-size N  指定批大小 (默认: 16)
  --timeout N   指定超时时间(秒)
  --parallel    并行执行(当支持时)
  --output DIR  指定输出目录
  --baseline V  指定基线版本进行对比
  --help        显示此帮助信息

示例:
  $0 --level L2                    # 运行模型验证层
  $0 --quick                       # 快速验证(L0+L1)
  $0 --all --baseline v1.0.0       # 全面验证并与基线对比
  $0 --level L3 --batch-size 32    # 性能验证，指定批大小

EOF
}

# 默认参数
LEVELS=()
RUN_ALL=false
QUICK_MODE=false
NO_CUDA=false
BATCH_SIZE=16
TIMEOUT=""
PARALLEL=false
BASELINE=""
OUTPUT_DIR="${RESULTS_DIR}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LEVELS+=("$2")
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --no-cuda)
            NO_CUDA=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# 确定要运行的层级
if [ "$RUN_ALL" = true ]; then
    LEVELS=("L0" "L1" "L2" "L3")
elif [ "$QUICK_MODE" = true ]; then
    LEVELS=("L0" "L1")
elif [ ${#LEVELS[@]} -eq 0 ]; then
    LEVELS=("L0")  # 默认只运行L0
fi

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

log_info "ST-BIF 分层验证系统启动"
log_info "项目根目录: ${PROJECT_ROOT}"
log_info "输出目录: ${OUTPUT_DIR}"
log_info "验证层级: ${LEVELS[*]}"

# 环境检查
check_environment() {
    log_stage "环境检查"
    
    # Python环境
    if ! python --version &>/dev/null; then
        log_error "Python环境不可用"
        exit 1
    fi
    
    # CUDA检查
    if [ "$NO_CUDA" = false ]; then
        if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            log_warning "CUDA不可用，将跳过CUDA相关测试"
            NO_CUDA=true
        fi
    fi
    
    log_success "环境检查完成"
}

# L0: 快速验证层
run_L0_validation() {
    log_stage "L0 - 快速验证层 (1-2分钟)"
    
    local start_time=$(date +%s)
    local success=true
    
    # 运行快速验证脚本
    if ! "${CI_ROOT}/scripts/quick_validate.sh" &> "${OUTPUT_DIR}/L0_validation.log"; then
        success=false
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 记录结果
    cat > "${OUTPUT_DIR}/L0_result.json" << EOF
{
    "level": "L0",
    "status": "$([ "$success" = true ] && echo "PASS" || echo "FAIL")",
    "duration": ${duration},
    "timestamp": "$(date -Iseconds)"
}
EOF
    
    if [ "$success" = true ]; then
        log_success "L0验证通过 (${duration}秒)"
    else
        log_error "L0验证失败 (${duration}秒)"
        return 1
    fi
}

# L1: 算子验证层
run_L1_validation() {
    log_stage "L1 - 算子验证层 (5-10分钟)"
    
    local start_time=$(date +%s)
    local success=true
    
    if [ "$NO_CUDA" = true ]; then
        log_warning "跳过L1验证 (CUDA不可用)"
        return 0
    fi
    
    # CUDA算子等效性测试
    log_info "运行CUDA算子等效性测试..."
    if ! python neuron_cupy/test_snn_operator.py &> "${OUTPUT_DIR}/L1_cuda_operator.log"; then
        log_error "CUDA算子测试失败"
        success=false
    fi
    
    # CUDA内核对比测试
    log_info "运行CUDA内核对比测试..."
    if ! python profile/scripts/cuda_kernel_profiler.py --quick --batch-size "${BATCH_SIZE}" &> "${OUTPUT_DIR}/L1_kernel_comparison.log"; then
        log_error "CUDA内核对比失败"
        success=false
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 记录结果
    cat > "${OUTPUT_DIR}/L1_result.json" << EOF
{
    "level": "L1",
    "status": "$([ "$success" = true ] && echo "PASS" || echo "FAIL")",
    "duration": ${duration},
    "timestamp": "$(date -Iseconds)",
    "cuda_available": $([ "$NO_CUDA" = false ] && echo "true" || echo "false")
}
EOF
    
    if [ "$success" = true ]; then
        log_success "L1验证通过 (${duration}秒)"
    else
        log_error "L1验证失败 (${duration}秒)"
        return 1
    fi
}

# L2: 模型验证层
run_L2_validation() {
    log_stage "L2 - 模型验证层 (10-20分钟)"
    
    local start_time=$(date +%s)
    local success=true
    
    # 检查模型文件
    if [ ! -f "checkpoints/resnet/best_ANN.pth" ]; then
        log_error "ANN模型文件不存在: checkpoints/resnet/best_ANN.pth"
        return 1
    fi
    
    if [ ! -f "checkpoints/resnet/best_QANN.pth" ]; then
        log_error "QANN模型文件不存在: checkpoints/resnet/best_QANN.pth"
        return 1
    fi
    
    # 运行完整转换流水线测试
    log_info "运行ANN→QANN→SNN转换流水线..."
    if ! python examples/ann_to_snn_conversion.py --quiet --batch-size "${BATCH_SIZE}" &> "${OUTPUT_DIR}/L2_conversion_pipeline.log"; then
        log_error "模型转换流水线失败"
        success=false
    fi
    
    # 精度回归检测
    log_info "检查模型精度回归..."
    if [ -f "${CI_ROOT}/baselines/model_accuracy_baseline.json" ]; then
        # 实现精度对比逻辑（简化版）
        log_info "精度基线对比功能待实现"
    else
        log_warning "精度基线文件不存在，跳过回归检测"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 记录结果
    cat > "${OUTPUT_DIR}/L2_result.json" << EOF
{
    "level": "L2",
    "status": "$([ "$success" = true ] && echo "PASS" || echo "FAIL")",
    "duration": ${duration},
    "timestamp": "$(date -Iseconds)",
    "batch_size": ${BATCH_SIZE}
}
EOF
    
    if [ "$success" = true ]; then
        log_success "L2验证通过 (${duration}秒)"
    else
        log_error "L2验证失败 (${duration}秒)"
        return 1
    fi
}

# L3: 性能验证层
run_L3_validation() {
    log_stage "L3 - 性能验证层 (20-60分钟)"
    
    local start_time=$(date +%s)
    local success=true
    
    # 运行性能基准测试
    log_info "运行性能基准测试..."
    if ! bash profile/scripts/quick_profile.sh &> "${OUTPUT_DIR}/L3_performance_benchmark.log"; then
        log_error "性能基准测试失败"
        success=false
    fi
    
    # CUDA内核详细性能分析
    if [ "$NO_CUDA" = false ]; then
        log_info "运行CUDA内核详细性能分析..."
        if ! python profile/scripts/cuda_kernel_profiler.py --batch-size "${BATCH_SIZE}" &> "${OUTPUT_DIR}/L3_cuda_detailed.log"; then
            log_error "CUDA内核详细分析失败"
            success=false
        fi
    fi
    
    # 性能回归检测
    if [ -f "${CI_ROOT}/baselines/performance_baseline.json" ] && [ -n "$BASELINE" ]; then
        log_info "运行性能回归检测..."
        # 实现性能对比逻辑
        log_info "性能基线对比功能待实现"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 记录结果
    cat > "${OUTPUT_DIR}/L3_result.json" << EOF
{
    "level": "L3",
    "status": "$([ "$success" = true ] && echo "PASS" || echo "FAIL")",
    "duration": ${duration},
    "timestamp": "$(date -Iseconds)",
    "baseline": "${BASELINE}"
}
EOF
    
    if [ "$success" = true ]; then
        log_success "L3验证通过 (${duration}秒)"
    else
        log_error "L3验证失败 (${duration}秒)"
        return 1
    fi
}

# L4: 深度验证层
run_L4_validation() {
    log_stage "L4 - 深度验证层 (小时级)"
    
    log_warning "L4深度验证层尚未实现"
    log_info "将包括: 长时间稳定性测试、内存泄漏检测、压力测试"
    
    # 占位符实现
    cat > "${OUTPUT_DIR}/L4_result.json" << EOF
{
    "level": "L4",
    "status": "SKIP",
    "duration": 0,
    "timestamp": "$(date -Iseconds)",
    "note": "L4层功能尚未实现"
}
EOF
    
    return 0
}

# 主验证流程
main() {
    local overall_start=$(date +%s)
    local failed_levels=()
    
    # 环境检查
    check_environment
    
    # 运行各层级验证
    for level in "${LEVELS[@]}"; do
        case $level in
            L0)
                if ! run_L0_validation; then
                    failed_levels+=("L0")
                fi
                ;;
            L1)
                if ! run_L1_validation; then
                    failed_levels+=("L1")
                fi
                ;;
            L2)
                if ! run_L2_validation; then
                    failed_levels+=("L2")
                fi
                ;;
            L3)
                if ! run_L3_validation; then
                    failed_levels+=("L3")
                fi
                ;;
            L4)
                if ! run_L4_validation; then
                    failed_levels+=("L4")
                fi
                ;;
            *)
                log_error "未知验证层级: $level"
                failed_levels+=("$level")
                ;;
        esac
    done
    
    local overall_end=$(date +%s)
    local total_duration=$((overall_end - overall_start))
    
    # 生成总结报告
    cat > "${OUTPUT_DIR}/validation_summary.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "levels_requested": [$(printf '"%s",' "${LEVELS[@]}" | sed 's/,$//') ],
    "levels_failed": [$(printf '"%s",' "${failed_levels[@]}" | sed 's/,$//') ],
    "total_duration": ${total_duration},
    "overall_status": "$([ ${#failed_levels[@]} -eq 0 ] && echo "PASS" || echo "FAIL")",
    "batch_size": ${BATCH_SIZE},
    "cuda_available": $([ "$NO_CUDA" = false ] && echo "true" || echo "false"),
    "baseline": "${BASELINE}"
}
EOF
    
    # 显示最终结果
    echo
    echo "========================================"
    log_stage "验证完成汇总"
    
    if [ ${#failed_levels[@]} -eq 0 ]; then
        log_success "所有验证层级通过！"
        log_success "验证层级: ${LEVELS[*]}"
        log_success "总耗时: ${total_duration}秒"
        exit 0
    else
        log_error "验证失败！"
        log_error "失败层级: ${failed_levels[*]}"
        log_error "总耗时: ${total_duration}秒"
        log_info "详细日志请查看: ${OUTPUT_DIR}/"
        exit 1
    fi
}

# 运行主流程
main