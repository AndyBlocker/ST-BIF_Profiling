# CLAUDE.md

此文件为Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个ST-BIF (Spike Threshold Bifurcation) 模块化脉冲神经网络框架，专注于高效的ANN到SNN转换。项目实现了完整的三阶段神经网络转换流水线，将传统人工神经网络转换为能耗优化的脉冲神经网络。

### 主要特点

- **模块化架构**: 清晰的代码组织和职责分离
- **ST-BIF神经元**: 先进的脉冲阈值分叉神经元模型
- **CUDA加速**: 自定义CUDA内核实现高性能脉冲计算
- **完整转换流水线**: ANN → QANN → SNN 无缝转换
- **向后兼容**: 保持与原始monolithic代码的完全兼容性

## 模块化架构

项目已重构为现代化的模块化结构：

```
ST-BIF_Profiling/
├── snn/                    # 核心SNN框架
│   ├── neurons/            # ST-BIF和IF神经元实现
│   │   ├── if_neurons.py   # 标准IF神经元
│   │   └── st_bif_neurons.py # ST-BIF神经元
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
│   └── misc.py            # 杂项工具
├── examples/              # 使用示例
│   └── ann_to_snn_conversion.py # 完整转换示例
├── checkpoints/           # 预训练模型
│   └── resnet/            # ResNet检查点
├── legacy/                # 原始实现（只读）
│   ├── spike_quan_layer.py
│   └── spike_quan_wrapper_ICML.py
└── neuron_cupy/           # CUDA加速脉冲神经元
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
- **支持**: 多精度（fp32, fp16）和不同神经元类型

## 开发指南

### 依赖安装
```bash
# 核心深度学习依赖
pip install torch torchvision

# CUDA加速（根据CUDA版本选择）
pip install cupy-cuda11x  # 或 cupy-cuda12x

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

### 目录和输出
- **检查点**: `checkpoints/resnet/`
- **输出目录**: `/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/`
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
2. 确保所有测试通过且精度一致
3. 保持代码风格和文档标准
4. 验证向后兼容性

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size到32或64
2. **模型文件缺失**: 检查`checkpoints/resnet/`路径
3. **导入错误**: 确保从项目根目录运行
4. **精度下降**: 验证模型转换参数设置

### 调试技巧
- 使用`--quiet`模式进行性能测试
- 检查CUDA设备可用性
- 验证模型文件完整性
- 比较模块化vs原始实现结果

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