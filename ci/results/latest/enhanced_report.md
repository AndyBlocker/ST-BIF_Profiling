# 🧪 ST-BIF CI/CD 测试报告

## 📋 基本信息

- **测试时间**: 2025-06-09 17:44:22
- **Git分支**: ci-infrastructure
- **Git提交**: `b2f8072cae49`

## ❌ 总体结果

- **状态**: 失败
- **成功率**: 50.0% (1/2)

## 🔍 测试套件详情

### ✅ Import Tests

- **状态**: 通过
- **执行时间**: 2.5秒
- **测试统计**: 19通过, 0失败

### ❌ Basic CUDA Tests

- **状态**: 失败
- **执行时间**: 1.7秒
- **测试统计**: 10通过, 9失败
- **失败测试**:
  - `test_different_dimensions_regression[64-32]`
  - `test_different_dimensions_regression[128-8]`
  - `test_different_dimensions_regression[32-32]`
  - `test_different_dimensions_regression[64-8]`
  - `test_different_dimensions_regression[128-32]`
