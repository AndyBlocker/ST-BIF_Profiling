# ST-BIF SNN NVTX Profiling Guide

## 生成的文件

### 1. 主要脚本
- `nsys_snn_profiling.py` - 主要profiling脚本，包含详细的NVTX标记
- `run_nsys_profiling.sh` - nsys执行脚本

### 2. 输出文件
- `snn_profile_YYYYMMDD_HHMMSS.nsys-rep` - nsys GUI分析文件
- `nsys_profiling_results_YYYYMMDD_HHMMSS.json` - 详细timing数据
- `nsys_profiling_summary_YYYYMMDD_HHMMSS.txt` - 人类可读摘要

## 使用方法

### 方法1：完整nsys profiling（推荐）
```bash
# 运行nsys profiling
./run_nsys_profiling.sh

# 或者手动运行
nsys profile -o snn_profile python nsys_snn_profiling.py
```

### 方法2：仅timing数据（测试用）
```bash
# 不用nsys，仅获取timing数据
python nsys_snn_profiling.py
```

## NVTX标记层次结构

```
Complete_Profiling_Session
├── Model_Loading
│   ├── QANN_Creation
│   └── SNN_Wrapper_Creation
├── Warmup_Runs
│   ├── Warmup_0
│   └── Warmup_1
└── Main_Inference_Timing_Run_X (X = 0,1,2,3,4)
    ├── Model_Reset
    ├── Time_Encoding
    ├── Input_Reshaping
    ├── Model_Forward_Pass
    │   └── Model_Compute_All_Layers
    ├── Output_Processing
    ├── Memory_Profiling_Run_X
    └── Detailed_Layer_Profiling_Run_X (仅前2次)
        ├── Conv1_Block
        ├── MaxPool
        ├── Layer1_ResBlocks
        ├── Layer2_ResBlocks
        ├── Layer3_ResBlocks
        ├── Layer4_ResBlocks
        └── Final_Layers
```

## 关键配置参数

### 当前设置
- **Batch size**: 32
- **推理次数**: 5次
- **预热次数**: 2次
- **时间步**: 8 (T=8)
- **神经元类型**: ST-BIF
- **量化级别**: 8

### 生成的数据
- **总推理时间**: ~6.4ms (93.2%是Model_Forward_Pass)
- **组件分解**: Reset(3.4%) + Encoding(2.5%) + Forward(93.2%) + Output(0.3%)
- **层级分解**: Layer3/Layer4最耗时(~1.86ms each)
- **内存使用**: ~183MB peak

## nsys GUI分析重点

### 1. 查看整体时间线
- 在Timeline中查看`Complete_Profiling_Session`
- 观察5次运行的一致性

### 2. 分析主要瓶颈
- 重点查看`Model_Forward_Pass`内的CUDA kernels
- 观察`Model_Compute_All_Layers`中的kernel分布

### 3. 层级分析
- 在前2次运行中查看`Detailed_Layer_Profiling_Run_X`
- 比较不同ResNet layer的性能

### 4. CUDA kernel分析
- 查看ST-BIF相关的CUDA kernels
- 分析kernel launch frequency和duration
- 观察memory transfer patterns

### 5. 内存分析  
- 查看GPU memory usage timeline
- 观察内存分配patterns

## 典型分析工作流

1. **打开nsys文件**
   ```bash
   nsight-sys snn_profile_YYYYMMDD_HHMMSS.nsys-rep
   ```

2. **缩放到Complete_Profiling_Session范围**

3. **分析5次推理的一致性**
   - 时间是否稳定
   - kernel pattern是否相同

4. **深入Model_Forward_Pass**
   - 找出最耗时的kernels
   - 分析kernel之间的gap

5. **对比层级性能**
   - Layer3/Layer4为什么最慢
   - Conv1 block的性能特征

6. **识别优化机会**
   - Kernel launch overhead
   - Memory transfer bottlenecks
   - Underutilized GPU time

## 预期发现

基于初步数据，您可能会发现：

1. **ST-BIF CUDA kernels**占主导地位
2. **Layer3/Layer4**的ResNet blocks最耗时
3. **Kernel launch overhead**可能显著
4. **Memory access patterns**可能不够优化
5. **GPU utilization**可能有提升空间

## 下一步分析

1. 识别具体的ST-BIF kernel名称和调用频率
2. 分析kernel的grid/block配置
3. 查看memory bandwidth utilization
4. 识别potential kernel fusion机会
5. 分析batch processing efficiency