# ST-BIF CI/CD 快速开始指南

## 🚀 立即可用的CI命令

### 日常开发验证
```bash
# 快速验证 (1-2分钟) - 推荐日常使用
./ci/scripts/quick_validate.sh

# 快速CI流水线 (3-5分钟)
python ci/scripts/ci_runner.py --mode quick
```

### 提交前检查
```bash
# 标准CI验证 (10-15分钟)
python ci/scripts/ci_runner.py --mode standard

# 完整验证 (20-30分钟)
python ci/scripts/ci_runner.py --mode full
```

### 基线管理
```bash
# 为当前版本创建基线快照
python ci/scripts/generate_baseline.py --output ci/baselines/v1.0.0_current

# 更新基线并验证
python ci/scripts/ci_runner.py --mode baseline
```

### 专项测试
```bash
# CUDA内核等效性测试
python ci/scripts/cuda_equivalence_guard.py

# 回归测试套件
python ci/scripts/regression_test_suite.py --baseline ci/baselines/v1.0.0_current

# 分层验证
./ci/scripts/run_validation.sh --level L2
```

## 📊 CI结果解读

### 快速验证结果
- ✅ **全部通过**: 代码可以安全提交
- ⚠️ **部分警告**: 检查具体警告信息，通常可以继续
- ❌ **测试失败**: 必须修复后再提交

### 常见失败原因和解决方案

#### 1. 导入错误
```
ModuleNotFoundError: No module named 'neuron_cupy'
```
**解决**: 确保从项目根目录运行，检查Python路径

#### 2. CUDA内核等效性失败
```
CUDA内核等效性验证失败
```
**解决**: 检查新版CUDA内核实现，查看详细差异报告

#### 3. 模型精度回归
```
检测到精度回归: SNN精度下降超过阈值
```
**解决**: 检查模型转换流程，验证参数设置

#### 4. 性能回归
```
CUDA性能测试执行失败
```
**解决**: 检查GPU环境，确保CUDA环境正常

## 🛠️ 开发工作流集成

### Git Hook 集成 (推荐)
```bash
# 创建pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "运行ST-BIF快速验证..."
./ci/scripts/quick_validate.sh
if [ $? -ne 0 ]; then
    echo "❌ CI验证失败，请修复后再提交"
    exit 1
fi
echo "✅ CI验证通过"
EOF

chmod +x .git/hooks/pre-commit
```

### IDE集成
在VS Code中添加任务配置 (`.vscode/tasks.json`):
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "ST-BIF Quick Validate",
            "type": "shell",
            "command": "./ci/scripts/quick_validate.sh",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        }
    ]
}
```

## 📁 CI结果文件位置

```
ci/
├── results/latest/          # 最新运行结果
│   ├── ci_results.json      # CI运行摘要
│   ├── quick_validation_summary.json
│   └── regression_test_results.json
├── baselines/               # 基线版本
│   ├── v1.0.0_current/      # 当前基线
│   └── v1.0.0_20250609/     # 历史基线
└── scripts/                 # CI脚本
```

## 🎯 性能基准

### 典型执行时间
- **快速验证**: 15-30秒
- **标准CI**: 5-10分钟  
- **完整CI**: 15-30分钟
- **基线生成**: 10-15分钟

### 系统要求
- **最低要求**: Python 3.8+, 4GB RAM
- **推荐配置**: Python 3.9+, CUDA GPU, 8GB+ RAM
- **CUDA**: 可选，但强烈推荐用于完整测试

## 🔧 故障排除

### 常见问题速查

**问题**: `permission denied`
**解决**: `chmod +x ci/scripts/*.sh`

**问题**: `CUDA out of memory`  
**解决**: 使用更小的batch size: `--batch-size 16`

**问题**: `timeout`
**解决**: 增加超时时间或检查系统负载

**问题**: 基线文件不存在
**解决**: 运行 `python ci/scripts/generate_baseline.py`

### 获取帮助
```bash
# 查看脚本帮助
python ci/scripts/ci_runner.py --help
./ci/scripts/run_validation.sh --help

# 查看CI配置
cat ci/configs/validation_config.yaml
```

## 🎉 成功集成确认

运行以下命令确认CI系统正常工作:
```bash
# 1. 快速验证
./ci/scripts/quick_validate.sh

# 2. 生成基线
python ci/scripts/generate_baseline.py --output ci/baselines/test

# 3. 运行CI
python ci/scripts/ci_runner.py --mode quick

# 如果都成功，说明CI系统已就绪！
```

---

**下一步**: 集成到GitHub Actions或本地CI系统，设置自动化触发规则。