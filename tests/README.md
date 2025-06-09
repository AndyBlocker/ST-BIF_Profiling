# ST-BIF Testing Framework

基于pytest的现代化测试框架，集成了CUDA内核性能基准测试和CI/CD流水线。

## 🚀 快速开始

### 运行所有测试
```bash
# 运行所有测试
pytest

# 仅运行导入测试
pytest tests/test_imports.py

# 运行CUDA内核测试（需要GPU）
pytest tests/test_cuda_kernels.py -m cuda

# 运行模型转换测试
pytest tests/test_model_conversion.py
```

### 运行性能基准测试
```bash
# 基本性能测试
pytest tests/benchmark_cuda_kernels.py -m performance

# 完整基准测试（需要更长时间）
python tests/benchmark_cuda_kernels.py --quick

# 详细基准测试
python tests/benchmark_cuda_kernels.py
```

### 集成CI运行
```bash
# 快速CI验证（推荐日常使用）
python ci/scripts/run_pytest_ci.py --mode quick

# 标准CI验证（提交前检查）
python ci/scripts/run_pytest_ci.py --mode standard

# 完整CI验证（发布前检查）
python ci/scripts/run_pytest_ci.py --mode full
```

## 📋 测试分类

### 按功能分类
- **导入测试** (`test_imports.py`): 验证所有模块可以正确导入
- **CUDA内核测试** (`test_cuda_kernels.py`): 验证CUDA算子的正确性和性能
- **模型转换测试** (`test_model_conversion.py`): 验证ANN→QANN→SNN转换流水线
- **性能基准测试** (`benchmark_cuda_kernels.py`): 详细的性能对比分析

### 按标记分类
```bash
# 按标记运行测试
pytest -m "cuda"          # 需要CUDA的测试
pytest -m "equivalence"   # 等效性验证测试
pytest -m "performance"   # 性能测试
pytest -m "regression"    # 回归测试
pytest -m "slow"          # 耗时较长的测试
pytest -m "not slow"      # 快速测试

# 组合标记
pytest -m "cuda and not slow"      # CUDA测试但不包括耗时测试
pytest -m "equivalence and cuda"   # CUDA等效性测试
```

## 🔧 测试配置

### Pytest配置文件
配置文件: `pytest.ini`
- 自动发现测试文件
- 配置输出格式和详细程度
- 设置超时和并发选项
- 定义测试标记

### 测试Fixture
文件: `tests/conftest.py`
- 共享的测试设备配置
- 标准测试数据生成
- 通用断言函数
- 自动跳过CUDA测试（如果GPU不可用）

## 📊 性能基准测试

### CUDA内核对比
`tests/benchmark_cuda_kernels.py` 提供：
- 原始vs新版CUDA内核性能对比
- 多种配置下的详细基准测试
- 自动生成性能对比图表
- JSON格式的详细结果输出

### 基准测试结果
结果保存在: `tests/benchmark_results/`
- `benchmark_results_YYYYMMDD_HHMMSS.json`: 详细数据
- `performance_comparison_YYYYMMDD_HHMMSS.png`: 可视化图表

### 典型使用场景
```bash
# 快速性能检查
python tests/benchmark_cuda_kernels.py --quick

# 完整性能基准（推荐用于CI）
python tests/benchmark_cuda_kernels.py --runs 20

# 指定输出目录
python tests/benchmark_cuda_kernels.py --output my_results/
```

## 🔄 CI集成

### CI运行器
`ci/scripts/run_pytest_ci.py` 集成了pytest测试到ST-BIF CI框架中：

#### 快速模式（1-3分钟）
```bash
python ci/scripts/run_pytest_ci.py --mode quick
```
- 导入测试
- 基础CUDA功能测试

#### 标准模式（5-10分钟）
```bash
python ci/scripts/run_pytest_ci.py --mode standard
```
- 所有快速测试
- CUDA等效性测试
- 模型转换测试（不包括耗时测试）

#### 完整模式（20-60分钟）
```bash
python ci/scripts/run_pytest_ci.py --mode full
```
- 所有测试
- 性能基准测试
- 长时间稳定性测试

### CI结果
结果保存在: `ci/results/latest/`
- `pytest_ci_results.json`: CI运行摘要
- `pytest-results.xml`: JUnit格式测试结果

## 🧪 测试开发指南

### 添加新测试
1. 创建测试文件: `tests/test_your_feature.py`
2. 使用适当的pytest标记
3. 利用`conftest.py`中的共享fixture
4. 添加适当的跳过条件（如CUDA依赖）

### 测试最佳实践
- 使用描述性的测试名称
- 添加适当的pytest标记
- 处理可选依赖（GPU、特定库等）
- 使用参数化测试覆盖多种配置
- 提供清晰的错误消息

### 示例测试结构
```python
import pytest
import torch

class TestYourFeature:
    @pytest.mark.cuda
    @pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
    def test_your_cuda_feature(self, device, shape):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        
        # Your test code here
        pass
```

## 🐛 故障排除

### 常见问题

**问题**: `ModuleNotFoundError`
**解决**: 确保从项目根目录运行pytest

**问题**: CUDA测试失败
**解决**: 检查GPU可用性和CUDA环境

**问题**: 导入错误
**解决**: 检查依赖是否安装，路径是否正确

**问题**: 性能测试超时
**解决**: 调整`pytest.ini`中的timeout设置

### 调试技巧
```bash
# 详细输出
pytest -v -s

# 停在第一个失败
pytest -x

# 仅运行失败的测试
pytest --lf

# 详细回溯信息
pytest --tb=long
```

## 📈 集成到开发工作流

### Git Hooks
```bash
# 创建pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "运行快速测试..."
python ci/scripts/run_pytest_ci.py --mode quick
EOF

chmod +x .git/hooks/pre-commit
```

### IDE集成
在VS Code中添加任务:
```json
{
    "label": "Run Quick Tests",
    "type": "shell", 
    "command": "python ci/scripts/run_pytest_ci.py --mode quick",
    "group": "test"
}
```

---

这个测试框架为ST-BIF项目提供了全面、现代化的测试基础设施，支持从快速验证到详细性能分析的各种需求。