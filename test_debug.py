#!/usr/bin/env python3
"""
调试测试脚本 - 分析QANN性能问题
"""

import torch
import torch.nn as nn
import copy
from torchvision import datasets, transforms
from tqdm import tqdm
import time

# 导入必要的模块
from spike_quan_wrapper_ICML import myquan_replace_resnet
import resnet


def build_test_dataset():
    """构建测试数据集"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    testset = datasets.CIFAR10(root='/home/zilingwei/cifar10', train=False, download=True, transform=test_transform)
    return testset


def create_test_dataloader(dataset, batch_size=128):
    """创建测试数据加载器"""
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    return test_loader


def test_model_detailed(model, test_loader, model_name="Model"):
    """详细测试模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    print(f"\n🧪 详细测试{model_name}模型...")
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"测试{model_name}", unit="batch")
        
        for batch_idx, (data, target) in enumerate(test_pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 计算推理时间
            torch.cuda.synchronize()  # 确保同步
            start_time = time.time()
            output = model(data)
            torch.cuda.synchronize()  # 确保同步
            inference_time = time.time() - start_time
            total_time += inference_time
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            current_acc = 100. * correct / total
            test_pbar.set_postfix({
                'Acc': f'{current_acc:.2f}%',
                'Time': f'{inference_time*1000:.1f}ms'
            })
            
            # 只测试前几个batch来快速调试
            if batch_idx >= 10:  # 只测试前10个batch
                break
    
    accuracy = 100. * correct / total
    avg_time_per_batch = total_time / min(11, len(test_loader))
    avg_time_per_sample = total_time / total
    
    print(f"✅ {model_name}测试完成 (前{min(11, len(test_loader))}个batch):")
    print(f"   📊 准确率: {accuracy:.2f}%")
    print(f"   ⏱️  平均每batch时间: {avg_time_per_batch*1000:.2f}ms")
    print(f"   🚀 平均每样本时间: {avg_time_per_sample*1000:.3f}ms")
    
    return accuracy, avg_time_per_sample


def load_model_weights(model, model_path, model_name):
    """加载模型权重"""
    try:
        print(f"📂 加载{model_name}模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ {model_name}模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 加载{model_name}模型失败: {e}")
        return False


def debug_quantization_levels():
    """调试不同量化级别的影响"""
    print("\n" + "="*60)
    print("🔬 调试量化级别对性能的影响")
    print("="*60)
    
    # 准备数据
    test_set = build_test_dataset()
    test_loader = create_test_dataloader(test_set, 128)
    
    # 加载基础ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    if not load_model_weights(ann_model, "best_ANN.pth", "ANN"):
        print("❌ 无法加载基础ANN模型")
        return
    
    # 测试原始ANN
    print("\n📊 测试基础ANN模型")
    ann_acc, ann_time = test_model_detailed(ann_model, test_loader, "ANN")
    
    # 测试不同的量化级别
    levels = [4, 8, 16, 32]
    
    for level in levels:
        print(f"\n📊 测试Level {level}量化")
        
        # 深度复制模型
        qann_model = copy.deepcopy(ann_model)
        
        # 应用量化
        print(f"🔄 应用Level {level}量化...")
        myquan_replace_resnet(qann_model, level=level, weight_bit=32, is_softmax=False)
        
        # 测试量化模型
        qann_acc, qann_time = test_model_detailed(qann_model, test_loader, f"QANN-L{level}")
        
        # 计算性能变化
        acc_drop = ann_acc - qann_acc
        speed_ratio = ann_time / qann_time
        
        print(f"   📈 准确率变化: {ann_acc:.2f}% → {qann_acc:.2f}% ({acc_drop:+.2f}%)")
        print(f"   🚀 速度变化: {speed_ratio:.2f}x")
        
        # 清理内存
        del qann_model
        torch.cuda.empty_cache()


def debug_model_structure():
    """调试模型结构变化"""
    print("\n" + "="*60)
    print("🔬 调试模型结构变化")
    print("="*60)
    
    # 创建基础模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    print("📋 原始ANN模型结构:")
    print("relu层数量:", sum(1 for m in ann_model.modules() if isinstance(m, nn.ReLU)))
    
    # 应用量化
    qann_model = copy.deepcopy(ann_model)
    myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    
    print("\n📋 量化后QANN模型结构:")
    from spike_quan_layer import MyQuan
    print("MyQuan层数量:", sum(1 for m in qann_model.modules() if isinstance(m, MyQuan)))
    print("ReLU层数量:", sum(1 for m in qann_model.modules() if isinstance(m, nn.ReLU)))
    
    # 检查量化层的参数
    print("\n🔍 量化层参数:")
    for name, module in qann_model.named_modules():
        if isinstance(module, MyQuan):
            print(f"  {name}: level={module.pos_max}, threshold={module.s.item():.4f}")


def main():
    print("🔬 QANN性能调试脚本")
    print(f"📱 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 调试1: 模型结构变化
    debug_model_structure()
    
    # 调试2: 不同量化级别的影响  
    debug_quantization_levels()
    
    print("\n🎉 调试完成！")


if __name__ == "__main__":
    main()