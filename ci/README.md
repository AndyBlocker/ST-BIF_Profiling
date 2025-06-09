# ST-BIF CI/CD Framework

这是ST-BIF项目的持续集成和持续部署框架，提供多层级的验证体系，确保代码质量和性能稳定性。

## 🎯 设计目标

- **质量保证**: 确保每次变更不会破坏模型精度或系统稳定性
- **性能守护**: 防止性能回归，维护CUDA算子优化效果  
- **开发效率**: 提供快速反馈，支持增量验证和本地测试
- **协作支持**: 标准化验证流程，便于多人协作开发

## 🏗️ 框架架构

### 分层验证体系

```
L4 [Deep]     │ 长时间稳定性测试、内存泄漏检测 (小时级)
L3 [Perf]     │ 性能基准测试、回归检测 (20-60分钟) 
L2 [Model]    │ 模型转换流水线、精度验证 (10-20分钟)
L1 [Operator] │ CUDA内核验证、数值等效性 (5-10分钟)
L0 [Quick]    │ 语法检查、导入测试、单元测试 (1-2分钟)
```

### 触发策略

| 层级 | 本地开发 | PR提交 | 主分支合并 | 发布前 |
|------|----------|--------|------------|---------|
| L0   | ✅ 每次修改 | ✅ 必须 | ✅ 必须 | ✅ 必须 |
| L1   | ✅ 算子修改 | ✅ 必须 | ✅ 必须 | ✅ 必须 |
| L2   | 🔶 模型修改 | ✅ 必须 | ✅ 必须 | ✅ 必须 |
| L3   | 🔶 性能关注 | 🔶 可选 | ✅ 必须 | ✅ 必须 |
| L4   | ❌ 手动 | ❌ 跳过 | 🔶 可选 | ✅ 必须 |

## 📋 验证检查清单

### L0 - 快速验证层 (1-2分钟)
- [ ] 代码风格检查 (flake8, black)
- [ ] 静态类型检查 (mypy)
- [ ] 导入完整性测试
- [ ] 基础单元测试
- [ ] 文档生成验证

### L1 - 算子验证层 (5-10分钟)  
- [ ] CUDA算子数值等效性验证
- [ ] 多精度算子测试 (fp16/fp32/fp64)
- [ ] 梯度计算一致性检查
- [ ] 边界条件测试
- [ ] 内存访问安全检查

### L2 - 模型验证层 (10-20分钟)
- [ ] ANN→QANN→SNN完整转换流水线
- [ ] 模型精度回归检测 (阈值: ±0.5%)
- [ ] 检查点加载和保存验证
- [ ] 多配置兼容性测试
- [ ] 向后兼容性检查

### L3 - 性能验证层 (20-60分钟)
- [ ] 推理性能基准测试
- [ ] CUDA内核性能对比
- [ ] 内存使用分析
- [ ] GPU利用率检查
- [ ] 吞吐量回归检测

### L4 - 深度验证层 (小时级)
- [ ] 长时间稳定性测试 (1000+ iterations)
- [ ] 内存泄漏检测
- [ ] 并发访问测试
- [ ] 极限负载测试
- [ ] 故障恢复测试

## 🔧 本地使用

### 快速验证 (推荐日常使用)
```bash
# 运行快速验证套件
./ci/scripts/quick_validate.sh

# 仅验证当前修改的模块
./ci/scripts/incremental_validate.sh --changed-only
```

### 完整验证
```bash
# 运行指定层级的验证
./ci/scripts/run_validation.sh --level L2

# 运行所有层级验证
./ci/scripts/run_validation.sh --all

# 性能对比分析
./ci/scripts/performance_comparison.sh --baseline v1.0.0
```

### 开发者工作流
```bash
# 1. 开发过程中的快速反馈
./ci/scripts/quick_validate.sh

# 2. 提交前的完整检查
./ci/scripts/pre_commit_check.sh

# 3. 性能影响评估
./ci/scripts/performance_impact.sh --compare-to HEAD~1
```

## 📊 结果报告

### 输出目录结构
```
ci/results/
├── latest/                    # 最新运行结果
├── baselines/                 # 基线版本结果
├── history/                   # 历史运行记录
└── reports/                   # 生成的报告
    ├── performance_trends.png
    ├── accuracy_comparison.json
    └── validation_summary.html
```

### 关键指标跟踪
- **模型精度**: ANN/QANN/SNN各阶段精度
- **性能指标**: 推理时间、吞吐量、内存使用
- **CUDA内核**: 延迟、数值误差、GPU利用率
- **稳定性**: 成功率、错误模式、资源消耗

## ⚙️ 配置管理

### 主配置文件
- `ci/configs/validation_config.yaml`: 验证参数配置
- `ci/configs/performance_thresholds.yaml`: 性能阈值设置
- `ci/configs/test_environments.yaml`: 测试环境配置

### 环境适配
- **本地开发**: 自动检测可用GPU和CUDA版本
- **CI服务器**: 标准化GPU环境配置
- **多GPU支持**: 并行测试加速

## 🚀 扩展指南

### 添加新的验证层级
1. 在`ci/scripts/validators/`创建验证脚本
2. 更新`ci/configs/validation_config.yaml`
3. 添加到主验证流水线`ci/scripts/run_validation.sh`

### 集成新的性能指标
1. 在`ci/scripts/metrics/`添加指标收集脚本
2. 更新基线生成器`ci/scripts/generate_baseline.sh`
3. 扩展报告生成器`ci/scripts/generate_reports.py`

### 支持新的模型架构
1. 在`ci/configs/test_models.yaml`添加模型配置
2. 扩展模型测试脚本`ci/scripts/validators/model_validator.py`
3. 更新性能基准`ci/baselines/`

## 📞 支持和维护

### 常见问题
- **GPU内存不足**: 调整batch size配置
- **CUDA版本不匹配**: 检查环境配置
- **性能基线丢失**: 重新生成基线

### 维护计划  
- **每周**: 性能趋势分析
- **每月**: 基线更新和阈值调整
- **每季度**: 框架功能扩展评估

### 联系方式
- 技术问题: 创建GitHub Issue
- 功能建议: 提交Pull Request
- 紧急问题: 联系项目维护者