# CLAUDE.md

此文件为Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个ST-BIF (Spike Threshold Bifurcation) 模块化脉冲神经网络框架，专注于高效的ANN到SNN转换和性能优化分析。项目实现了完整的三阶段神经网络转换流水线，并提供了全面的性能分析工具套件，将传统人工神经网络转换为能耗优化的脉冲神经网络。

### 主要特点

- **模块化架构**: 清晰的代码组织和职责分离
- **ST-BIF神经元**: 先进的脉冲阈值分叉神经元模型
- **CUDA加速**: 自定义CUDA内核实现高性能脉冲计算
- **完整转换流水线**: ANN → QANN → SNN 无缝转换
- **专业性能分析**: 集成PyTorch Profiler、NSYS、NCU的完整分析工具链
- **CUDA内核优化**: 多版本内核对比分析和性能验证
- **向后兼容**: 保持与原始monolithic代码的完全兼容性

## 模块化架构

项目已重构为现代化的模块化结构：

```
ST-BIF_Profiling/
├── snn/                    # 核心SNN框架
│   ├── neurons/            # ST-BIF和IF神经元实现
│   │   ├── if_neurons.py   # 标准IF神经元
│   │   ├── st_bif_neurons.py # ST-BIF神经元
│   │   └── st_bif_neurons_profiled.py # 带NVTX性能标记的ST-BIF神经元
│   ├── layers/             # SNN专用层
│   │   ├── quantization.py # 量化层
│   │   ├── conv.py         # 脉冲卷积层
│   │   ├── linear.py       # 脉冲全连接层
│   │   ├── normalization.py # 脉冲归一化层
│   │   └── pooling.py      # 脉冲池化层
│   └── conversion/         # 模型转换工具
│       └── quantization.py # ANN→QANN转换函数
├── models/                 # 神经网络模型
│   └── resnet.py          # ResNet实现
├── wrapper/                # SNN包装器
│   ├── snn_wrapper.py      # 主要SNN包装类
│   ├── encoding.py         # 时间编码工具
│   ├── reset.py            # 神经元重置工具
│   ├── attention_conversion.py # 注意力层转换
│   └── base.py             # 基础类
├── utils/                  # 框架工具函数
│   ├── functions.py        # 通用功能函数
│   ├── io.py              # 输入输出工具
│   ├── misc.py            # 杂项工具
│   └── reset_fast.py      # 快速重置工具
├── examples/              # 使用示例
│   └── ann_to_snn_conversion.py # 完整转换示例
├── checkpoints/           # 预训练模型
│   └── resnet/            # ResNet检查点
├── legacy/                # 原始实现（只读）
│   ├── spike_quan_layer.py
│   └── spike_quan_wrapper_ICML.py
├── neuron_cupy/           # CUDA加速脉冲神经元
│   ├── cuda_operator.py   # 原始CUDA算子
│   ├── cuda_operator_new.py # 新版CUDA算子实现
│   ├── cuda_snn_kernels.cu # 原始CUDA内核
│   └── cuda_snn_kernels_new.cu # 新版CUDA内核
└── profile/               # 性能分析工具套件
    ├── README.md          # 性能分析概述
    ├── USAGE.md           # 详细使用指南
    ├── NSYS_PROFILING_GUIDE.md # NSYS分析指南
    ├── configs/           # 配置文件
    │   └── profile_config.yaml
    ├── outputs/           # 分析结果输出
    │   └── nsys_results/  # NSYS分析结果
    └── scripts/           # 性能分析脚本
        ├── ann_profiler.py # ANN基线性能分析
        ├── snn_profiler.py # SNN性能分析
        ├── cuda_kernel_profiler.py # CUDA内核对比分析
        ├── cuda_kernel_benchmark.py # CUDA内核基准测试
        ├── nsys_snn_profiling.py # NSYS专用SNN分析
        ├── ncu_kernel_comparison.sh # NCU内核对比
        ├── compare_ann_snn.sh # ANN vs SNN性能对比
        ├── quick_profile.sh # 快速性能检查
        └── run_all_profiles.sh # 综合性能分析
```

### 三阶段转换流水线

1. **ANN (人工神经网络)**
   - 标准ResNet18架构
   - 在CIFAR-10上训练
   - 浮点精度计算

2. **QANN (量化人工神经网络)**
   - 应用可学习量化器
   - 使用`myquan_replace_resnet()`转换
   - 保持高精度同时减少计算需求

3. **SNN (脉冲神经网络)**
   - ST-BIF神经元替换ReLU激活
   - 时间编码输入处理
   - 事件驱动的稀疏计算

## 核心组件

### ST-BIF神经元
- **位置**: `snn/neurons/st_bif_neurons.py`
- **特点**: 
  - 可学习阈值
  - 分叉动力学
  - 多步时间处理
  - CUDA加速支持

### 模型转换
```python
from snn.conversion import myquan_replace_resnet
from wrapper import SNNWrapper_MS

# ANN → QANN
myquan_replace_resnet(model, level=8, weight_bit=32)

# QANN → SNN  
snn_model = SNNWrapper_MS(
    ann_model=qann_model,
    time_step=8,
    level=8,
    neuron_type="ST-BIF"
)
```

### CUDA加速
- **位置**: `neuron_cupy/`
- **功能**: 自定义CUDA内核实现高效脉冲计算
- **支持**: 多精度（fp32, fp16, fp64）和不同神经元类型
- **内核版本**: 
  - `cuda_operator.py` + `cuda_snn_kernels.cu`: 原始稳定版本
  - `cuda_operator_new.py` + `cuda_snn_kernels_new.cu`: 优化版本（Vec2向量化）

### 性能分析工具套件
- **位置**: `profile/`
- **功能**: 全面的GPU和CPU性能分析工具链
- **工具**: 
  - PyTorch Profiler: 原生PyTorch性能分析
  - CUDA内核对比: 多版本内核性能对比和等效性验证
  - NVIDIA Nsight Systems: GPU时间线和系统级分析
  - NVIDIA Nsight Compute: 详细内核级性能分析

## 开发指南

### 依赖安装
```bash
# 核心深度学习依赖
pip install torch torchvision

# CUDA加速（根据CUDA版本选择）
pip install cupy-cuda11x  # 或 cupy-cuda12x

# 性能分析依赖
pip install matplotlib pandas seaborn nvtx

# 其他依赖
pip install timm
```

### 运行示例
```bash
# 完整ANN→QANN→SNN转换示例
python examples/ann_to_snn_conversion.py

# 静默模式（用于基准测试）
python examples/ann_to_snn_conversion.py --quiet

# 自定义批大小
python examples/ann_to_snn_conversion.py --batch-size 64
```

### 性能分析快速开始
```bash
# 快速性能检查（推荐）
cd profile/scripts && ./quick_profile.sh

# CUDA内核对比分析
python profile/scripts/cuda_kernel_profiler.py

# 综合性能分析
./profile/scripts/run_all_profiles.sh
```

### 测试脉冲神经元
```bash
# 测试CUDA脉冲算子
python neuron_cupy/test_snn_operator.py
```

## 关键配置参数

- `level`: 量化级别（默认: 8）
- `time_step`/`T`: 时间编码步数（默认: 8）
- `weight_bit`: 权重精度位数（默认: 32）
- `neuron_type`: 神经元类型（"ST-BIF"）
- `Encoding_type`: 编码类型（"analog"）

## 性能特征

### 典型结果（CIFAR-10）
- **ANN精度**: ~86.74%
- **QANN精度**: ~85.17% (-1.57%)
- **SNN精度**: ~85.12% (-1.62%)
- **SNN速度**: ~0.45x ANN推理速度
- **能耗**: 显著降低（稀疏脉冲活动）

### CUDA内核性能对比
- **原始内核**: 稳定可靠，标量处理
- **新版内核**: Vec2向量化，奇数特征维度处理问题
- **等效性**: 新版内核在某些配置下数值差异较大
- **性能**: 不同配置下性能表现各有优劣

### 性能分析结果位置
- **性能分析输出**: `profile/outputs/`
- **CUDA内核对比图**: `profile/outputs/cuda_kernel_*/`
- **检查点**: `checkpoints/resnet/`
- **模型输出目录**: `/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/`
- **数据集**: `/home/zilingwei/cifar10`

## 代码维护规范

### Legacy文件保护
- **重要**: `legacy/spike_quan_layer.py`和`legacy/spike_quan_wrapper_ICML.py`是只读的原始实现
- **要求**: 新的模块化代码必须完整复刻原版功能
- **禁止**: 直接修改legacy文件
- **向后兼容**: 所有原有导入路径必须继续工作

### 模块化原则
- **单一职责**: 每个模块专注特定功能
- **清晰接口**: `__init__.py`只包含导入语句
- **文档完整**: 每个函数和类都有详细文档
- **测试覆盖**: 关键功能需要验证等效性

### 开发工作流
1. 使用`examples/ann_to_snn_conversion.py`验证功能
2. 运行`profile/scripts/quick_profile.sh`进行性能回归检测
3. 确保所有测试通过且精度一致
4. 进行CUDA内核对比分析（如有修改）
5. 保持代码风格和文档标准
6. 验证向后兼容性

### 性能分析工作流
1. **快速检查**: `cd profile/scripts && ./quick_profile.sh`
2. **内核对比**: `python profile/scripts/cuda_kernel_profiler.py`
3. **综合分析**: `./profile/scripts/run_all_profiles.sh`
4. **结果查看**: 检查`profile/outputs/`中的分析结果

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size到32或64
2. **模型文件缺失**: 检查`checkpoints/resnet/`路径
3. **导入错误**: 确保从项目根目录运行
4. **精度下降**: 验证模型转换参数设置
5. **性能分析失败**: 检查NVIDIA工具安装和CUDA环境
6. **内核对比错误**: 确保`cuda_snn_kernels_new.cu`文件存在

### 调试技巧
- 使用`--quiet`模式进行性能测试
- 检查CUDA设备可用性和内存使用
- 验证模型文件完整性
- 比较模块化vs原始实现结果
- 使用性能分析工具定位瓶颈
- 查看`profile/outputs/`中的详细分析报告
- 使用NVTX标记进行细粒度性能分析

## 扩展指南

### 添加新神经元类型
1. 在`layer/snn/neurons/`创建新文件
2. 继承基础神经元类
3. 实现前向传播和重置方法
4. 更新`__init__.py`导出

### 添加新层类型
1. 在`layer/snn/layers/`对应类别文件中添加
2. 遵循现有命名约定
3. 支持多步时间处理
4. 添加文档和示例

### 性能优化
- 利用CUDA内核加速
- 优化内存使用模式
- 实现稀疏计算优化
- 添加量化支持

### 性能分析工具扩展
1. 在`profile/scripts/`中添加新的分析脚本
2. 遵循现有命名约定（`*_profiler.py`或`*.sh`）
3. 更新`profile/README.md`和`profile/USAGE.md`
4. 添加配置参数到`profile/configs/profile_config.yaml`
5. 确保输出结果保存到`profile/outputs/`

### CUDA内核开发
1. 在`neuron_cupy/`中创建新的算子文件
2. 对应的`.cu`文件包含CUDA内核实现
3. 使用`cuda_kernel_profiler.py`进行性能验证
4. 确保数值等效性和性能提升
5. 更新文档说明新内核的特点和用途

## 重要提醒

### 性能分析最佳实践
- 使用`profile/scripts/quick_profile.sh`进行日常性能检查
- 定期运行CUDA内核对比分析，确保优化效果
- 查看可视化图表，重点关注延迟和内存使用趋势
- 在修改关键组件后必须进行综合性能分析

### CUDA内核修改注意事项
- 新内核(`cuda_operator_new.py`)目前存在等效性问题，谨慎使用
- 优先使用原始稳定内核(`cuda_operator.py`)用于生产环境
- 任何内核修改都必须通过`cuda_kernel_profiler.py`验证
- 关注Vec2向量化在奇数特征维度下的处理问题