#!/usr/bin/env python3
"""
快速测试脚本 - 分析QANN问题
"""

import torch
import torch.nn as nn
import copy
from torchvision import datasets, transforms
import time

# 导入必要的模块
from spike_quan_wrapper_ICML import myquan_replace_resnet
import resnet


def quick_test():
    """快速测试ANN vs QANN"""
    
    # 创建测试数据
    test_data = torch.randn(64, 3, 32, 32).cuda()
    
    # 创建ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    ann_model.cuda()
    ann_model.eval()
    
    print("🧠 ANN模型结构:")
    relu_count = sum(1 for m in ann_model.modules() if isinstance(m, nn.ReLU))
    print(f"   ReLU层数量: {relu_count}")
    
    # 测试ANN速度
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = ann_model(test_data)
        
        # 正式测试
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            output_ann = ann_model(test_data)
        torch.cuda.synchronize()
        ann_time = (time.time() - start_time) / 100
    
    print(f"   平均推理时间: {ann_time*1000:.3f}ms")
    print(f"   输出范围: [{output_ann.min().item():.3f}, {output_ann.max().item():.3f}]")
    
    # 创建QANN模型
    print(f"\n⚡ 转换为QANN (level=8)...")
    qann_model = copy.deepcopy(ann_model)
    myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    qann_model.eval()
    
    print("⚡ QANN模型结构:")
    from spike_quan_layer import MyQuan
    myquan_count = sum(1 for m in qann_model.modules() if isinstance(m, MyQuan))
    relu_count = sum(1 for m in qann_model.modules() if isinstance(m, nn.ReLU))
    print(f"   MyQuan层数量: {myquan_count}")
    print(f"   ReLU层数量: {relu_count}")
    
    # 测试QANN速度
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = qann_model(test_data)
        
        # 正式测试
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            output_qann = qann_model(test_data)
        torch.cuda.synchronize()
        qann_time = (time.time() - start_time) / 100
    
    print(f"   平均推理时间: {qann_time*1000:.3f}ms")
    print(f"   输出范围: [{output_qann.min().item():.3f}, {output_qann.max().item():.3f}]")
    
    # 比较结果
    print(f"\n📊 性能对比:")
    print(f"   速度变化: {ann_time/qann_time:.2f}x")
    print(f"   输出差异: {torch.abs(output_ann - output_qann).mean().item():.6f}")
    
    # 检查量化层参数
    print(f"\n🔍 量化层参数分析:")
    for name, module in qann_model.named_modules():
        if isinstance(module, MyQuan):
            print(f"   {name}: level={module.pos_max}, threshold={module.s.item():.4f}, sym={module.sym}")
            if hasattr(module, 'act_loss'):
                print(f"     act_loss={module.act_loss}")
            break  # 只显示第一个


if __name__ == "__main__":
    print("⚡ 快速QANN性能分析")
    print(f"📱 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    quick_test()
    
    print("\n🎉 分析完成！")