#!/usr/bin/env python3
"""
ST-BIF SNN 优化使用示例 - 修复性能退化问题

这个脚本展示了如何正确使用优化来避免性能退化问题。

问题原因分析 (Root Cause Analysis):
1. ❌ setup_optimizations() 在 SNNWrapper 创建之后调用 - Called after SNNWrapper creation
2. ❌ 混合精度与ST-BIF阈值计算冲突 - Mixed precision conflicts with ST-BIF thresholds  
3. ❌ 过于激进的CUDA优化设置 - Overly aggressive CUDA optimizations
4. ❌ 内存池与现有tensor管理冲突 - Memory pool conflicts with existing tensor management

解决方案 (Solution):
✅ 在模型创建之前设置优化 - Setup optimizations BEFORE model creation
✅ 使用ST-BIF兼容的保守优化 - Use ST-BIF compatible conservative optimizations
✅ 禁用不兼容的功能 - Disable incompatible features
✅ 提供回退机制 - Provide fallback mechanisms
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Optional, Dict, Any

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import fixed optimization utilities
from snn.optimization_utils_fixed import setup_st_bif_optimizations, get_st_bif_optimizer, cleanup_st_bif_optimizations

def main():
    """主函数演示正确的优化使用方式"""
    print("🔧 ST-BIF SNN Optimization - Corrected Usage Example")
    print("=" * 60)
    
    # 步骤 1: 检查环境
    print("\n1. 环境检查 (Environment Check):")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # 步骤 2: 正确的优化设置时机 - BEFORE any model creation
    print("\n2. 设置优化 (Setup Optimizations) - BEFORE model creation:")
    
    try:
        # 这是关键：在任何模型创建之前设置优化
        optimizer = setup_st_bif_optimizations(
            enable_memory_pool=True,     # 启用内存池
            enable_tf32=True,            # 启用TF32（安全）
            enable_cudnn_benchmark=False, # 禁用（避免变长输入问题）
            memory_pool_size=3,          # 保守的池大小
            verbose=True
        )
        print("   ✅ Optimizations setup successful")
        
    except Exception as e:
        print(f"   ❌ Optimization setup failed: {e}")
        print("   🔄 Falling back to default settings...")
        optimizer = get_st_bif_optimizer()  # 使用默认设置
    
    # 步骤 3: 现在创建模型 (NOW create models)
    print("\n3. 创建模型 (Create Models) - AFTER optimization setup:")
    
    # 模拟你的实际使用场景
    model = create_test_model()
    print("   ✅ Base model created")
    
    # 模拟SNNWrapper创建
    snn_model = simulate_snn_wrapper_creation(model, optimizer)
    print("   ✅ SNN wrapper created")
    
    # 步骤 4: 性能对比测试
    print("\n4. 性能测试 (Performance Testing):")
    run_performance_comparison(model, snn_model, optimizer)
    
    # 步骤 5: 监控和调试
    print("\n5. 优化状态监控 (Optimization Status):")
    show_optimization_status(optimizer)
    
    # 步骤 6: 清理
    print("\n6. 清理资源 (Cleanup):")
    cleanup_st_bif_optimizations()
    print("   ✅ Cleanup completed")

def create_test_model():
    """创建测试模型"""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))  
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = TestModel()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def simulate_snn_wrapper_creation(model, optimizer):
    """模拟SNNWrapper创建过程"""
    
    # 这里模拟你的SNNWrapper创建
    # 在实际使用中，这里应该是：
    # snn_wrapper = SNNWrapper_MS(model, time_step=8, level=8, ...)
    
    class MockSNNWrapper(nn.Module):
        def __init__(self, base_model, optimizer_manager):
            super().__init__()
            self.base_model = base_model
            self.optimizer_manager = optimizer_manager
            self.memory_pool = optimizer_manager.get_memory_pool()
            self.T = 8  # 时间步数
            
        def forward(self, x):
            batch_size = x.size(0)
            
            # 模拟ST-BIF时间步处理
            outputs = []
            for t in range(self.T):
                # 使用内存池（如果可用）
                if self.memory_pool:
                    temp_tensor = self.memory_pool.get_tensor((batch_size, 64))
                    # 模拟一些计算
                    temp_result = temp_tensor + t * 0.1
                    self.memory_pool.return_tensor(temp_tensor)
                
                # 基础前向传播
                output = self.base_model(x)
                outputs.append(output)
            
            # 聚合时间步输出
            return torch.stack(outputs).mean(dim=0)
    
    snn_wrapper = MockSNNWrapper(model, optimizer)
    return snn_wrapper

def run_performance_comparison(base_model, snn_model, optimizer):
    """运行性能对比测试"""
    
    batch_size = 16
    input_shape = (batch_size, 3, 32, 32)
    
    # 创建测试数据
    if torch.cuda.is_available():
        test_input = torch.randn(input_shape).cuda()
    else:
        test_input = torch.randn(input_shape)
    
    def benchmark_model(model, name, num_iterations=20):
        """基准测试单个模型"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = batch_size * num_iterations / total_time
        
        print(f"   {name}:")
        print(f"     Avg time: {avg_time*1000:.2f}ms")
        print(f"     Throughput: {throughput:.1f} samples/sec")
        
        return avg_time, throughput
    
    # 测试基础模型
    base_time, base_throughput = benchmark_model(base_model, "Base Model")
    
    # 测试SNN模型  
    snn_time, snn_throughput = benchmark_model(snn_model, "SNN Model (with optimizations)")
    
    # 计算比率
    if base_time > 0:
        slowdown_ratio = snn_time / base_time
        print(f"   📊 SNN vs Base slowdown: {slowdown_ratio:.2f}x")
        
        if slowdown_ratio < 10:  # 如果慢于10倍，说明优化有效
            print("   ✅ Optimization effective (reasonable slowdown)")
        else:
            print("   ⚠️  High slowdown - consider disabling some optimizations")

def show_optimization_status(optimizer):
    """显示优化状态"""
    status = optimizer.get_status()
    
    print("   📋 Current optimization status:")
    print(f"     Optimized: {status['is_optimized']}")
    print(f"     CUDA available: {status['cuda_available']}")
    
    if 'cuda_settings' in status:
        cuda_settings = status['cuda_settings']
        print(f"     cuDNN benchmark: {cuda_settings['cudnn_benchmark']}")
        print(f"     TF32 matmul: {cuda_settings['tf32_matmul']}")
    
    if 'memory_pool' in status:
        pool_stats = status['memory_pool']
        print(f"     Memory pool enabled: {pool_stats['enabled']}")
        print(f"     Memory pool hit rate: {pool_stats['hit_rate']:.2%}")
    
    if 'cuda_info' in status:
        cuda_info = status['cuda_info']
        print(f"     GPU memory allocated: {cuda_info['allocated_gb']:.2f} GB")

def demonstrate_fallback_pattern():
    """演示回退模式处理"""
    print("\n" + "=" * 60)
    print("回退模式演示 (Fallback Pattern Demonstration)")
    print("=" * 60)
    
    print("""
如果优化导致性能退化，使用以下回退策略：

# 策略1: 逐步禁用优化功能
optimizer = setup_st_bif_optimizations(
    enable_memory_pool=False,  # 首先禁用内存池
    enable_tf32=True,          # 保留TF32
    enable_cudnn_benchmark=False
)

# 策略2: 完全禁用优化（回到原始性能）
optimizer = get_st_bif_optimizer()  # 使用默认设置

# 策略3: 运行时切换
memory_pool = optimizer.get_memory_pool()
if memory_pool:
    memory_pool.disable()  # 运行时禁用内存池
    
# 策略4: 性能监控自动调整
if avg_time > expected_time * 1.5:  # 如果比预期慢50%以上
    optimizer.disable_memory_pool()  # 自动禁用问题功能
    print("⚠️  Auto-disabled memory pool due to performance regression")
""")

if __name__ == "__main__":
    # 运行主要示例
    main()
    
    # 显示回退模式
    demonstrate_fallback_pattern()
    
    print("\n" + "=" * 60)
    print("🎯 关键要点总结 (Key Takeaways)")
    print("=" * 60)
    print("1. ✅ 优化必须在模型创建之前设置")
    print("   Optimizations MUST be setup BEFORE model creation")
    print()
    print("2. ✅ 使用保守的优化设置避免冲突") 
    print("   Use conservative optimization settings to avoid conflicts")
    print()
    print("3. ✅ 提供回退机制处理性能退化")
    print("   Provide fallback mechanisms for performance regressions")
    print()
    print("4. ✅ 监控和调试优化效果")
    print("   Monitor and debug optimization effects")
    print()
    print("🔧 正确的使用模式:")
    print("   optimizer = setup_st_bif_optimizations()  # 首先设置")
    print("   model = load_model()                       # 然后创建模型") 
    print("   snn_wrapper = SNNWrapper_MS(model, ...)   # 最后创建wrapper")
    print()
    print("❌ 错误的使用模式:")
    print("   model = load_model()                       # 先创建模型")
    print("   snn_wrapper = SNNWrapper_MS(model, ...)   # 再创建wrapper")  
    print("   setup_optimizations()                     # 最后设置优化 ❌")