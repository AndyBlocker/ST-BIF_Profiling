# CLAUDE.md

此文件为Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目架构

这是一个专注于量化脉冲神经网络(SNNs)的深度学习研究项目，主要使用ResNet架构进行CIFAR-10分类。项目实现了三阶段神经网络转换流水线：

1. **ANN (人工神经网络)** - 标准ResNet模型
2. **QANN (量化人工神经网络)** - 可配置位精度的量化版本
3. **SNN (脉冲神经网络)** - 使用时间编码的事件驱动脉冲版本

### 核心组件

- **main.py**: 演示完整ANN→QANN→SNN转换流水线的入口点
- **resnet.py**: 为SNN转换修改的自定义ResNet实现
- **spike_quan_wrapper_ICML.py**: 处理模型转换和SNN功能的主要包装器
- **spike_quan_layer.py**: 量化和脉冲操作的自定义层（大文件~34k tokens）
- **neuron_cupy/**: 使用CuPy的CUDA加速脉冲神经元实现
- **glo.py**: 全局变量管理工具

### 关键架构特性

- **多精度支持**: FP32, FP16量化级别（通常level=8, weight_bit=32）
- **CUDA加速**: 脉冲神经元操作的自定义CUDA内核
- **时间步仿真**: 时间编码的可配置时间步数（T=8）
- **多种神经元模型**: ST-BIF（脉冲阈值分叉）神经元
- **注意力机制**: 支持transformer的量化和脉冲注意力机制

### 模型转换流程

1. 在CIFAR-10上训练标准ResNet18
2. 使用`myquan_replace_resnet()`应用量化
3. 使用`SNNWrapper_MS()`和时间编码转换为SNN
4. 每个阶段在降低计算需求的同时保持准确性

## 开发命令

由于没有找到包管理文件（requirements.txt, setup.py），依赖项可能需要手动安装：

```bash
# 核心依赖项（从导入推断）
pip install torch torchvision cupy-cuda11x timm

# 用于CUDA开发
# 确保安装CUDA工具包以进行自定义内核编译
```

### 运行项目

```bash
# 执行主要转换流水线
python main.py

# 测试脉冲神经元算子
python neuron_cupy/test_snn_operator.py
```

### CUDA内核开发

项目使用自定义CUDA内核进行加速脉冲操作：

- 内核位于`neuron_cupy/cuda_snn_kernels*.cu`
- 多精度变体（fp32, fp16）
- 针对不同神经元行为的修改版本

修改CUDA代码时，确保正确的内核编译和CuPy集成。

### 关键配置参数

- `level`: 量化级别（通常为8）
- `time_step`/`T`: 时间编码步数（通常为8）
- `weight_bit`: 权重精度（实验中为32）
- `neuron_type`: "ST-BIF"用于基于分叉的脉冲
- `Encoding_type`: "analog"用于连续输入编码

### 输出目录

模型输出和中间结果保存到：
`/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/`