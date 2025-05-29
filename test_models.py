#!/usr/bin/env python3
"""
模型测试脚本 - 只测试不训练
支持测试ANN、QANN、SNN三种模型的性能
"""

import torch
import torch.nn as nn
import os
import argparse
from torchvision import datasets, transforms
from tqdm import tqdm
import time

# 导入必要的模块
from spike_quan_wrapper_ICML import SNNWrapper_MS, myquan_replace_resnet
import resnet


def build_test_dataset():
    """构建测试数据集"""
    # CIFAR10的标准均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 测试数据预处理（不使用增强）
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


def test_model(model, test_loader, model_name="Model"):
    """测试模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    print(f"\n🧪 测试{model_name}模型...")
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"测试{model_name}", unit="batch")
        
        for data, target in test_pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 计算推理时间
            start_time = time.time()
            output = model(data)
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
    
    accuracy = 100. * correct / total
    avg_time_per_batch = total_time / len(test_loader)
    avg_time_per_sample = total_time / total
    
    print(f"✅ {model_name}测试完成:")
    print(f"   📊 准确率: {accuracy:.2f}%")
    print(f"   ⏱️  平均每batch时间: {avg_time_per_batch*1000:.2f}ms")
    print(f"   🚀 平均每样本时间: {avg_time_per_sample*1000:.3f}ms")
    print(f"   🎯 吞吐量: {1/avg_time_per_sample:.1f} samples/sec")
    
    return accuracy, avg_time_per_sample


def load_model_weights(model, model_path, model_name):
    """加载模型权重"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        print(f"📂 加载{model_name}模型: {model_path}")
        
        # 尝试加载checkpoint格式
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_acc' in checkpoint:
                print(f"   🏆 训练时最佳准确率: {checkpoint['best_acc']:.2f}%")
        else:
            # 直接加载state_dict
            model.load_state_dict(checkpoint)
        
        print(f"✅ {model_name}模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 加载{model_name}模型失败: {e}")
        return False


def test_ann_model(test_loader, model_path="best_ANN.pth"):
    """测试ANN模型"""
    print("\n" + "="*50)
    print("🧠 测试ANN模型")
    print("="*50)
    
    # 创建ANN模型
    model = resnet.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    # 加载权重
    if not load_model_weights(model, model_path, "ANN"):
        return None
    
    # 测试模型
    accuracy, inference_time = test_model(model, test_loader, "ANN")
    return {"accuracy": accuracy, "inference_time": inference_time}


def test_qann_model(test_loader, model_path="best_QANN.pth"):
    """测试QANN模型"""
    print("\n" + "="*50)
    print("⚡ 测试QANN模型")
    print("="*50)
    
    # 创建ANN模型
    model = resnet.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    # 转换为QANN
    print("🔄 转换为QANN...")
    myquan_replace_resnet(model, level=8, weight_bit=32, is_softmax=False)
    print("✅ 模型已转换为QANN")
    
    # 加载权重
    if not load_model_weights(model, model_path, "QANN"):
        return None
    
    # 测试模型
    accuracy, inference_time = test_model(model, test_loader, "QANN")
    return {"accuracy": accuracy, "inference_time": inference_time}


def test_snn_model(test_loader, model_path="best_SNN.pth"):
    """测试SNN模型"""
    print("\n" + "="*50)
    print("🧠 测试SNN模型")
    print("="*50)
    
    # 创建ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    # 转换为QANN
    print("🔄 转换为QANN...")
    myquan_replace_resnet(ann_model, level=8, weight_bit=32, is_softmax=False)
    
    # 转换为SNN
    print("🔄 转换为SNN...")
    output_dir = "/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/"
    os.makedirs(output_dir, exist_ok=True)
    
    snn_model = SNNWrapper_MS(
        ann_model=ann_model, 
        cfg=None, 
        time_step=8,
        Encoding_type="analog", 
        level=8, 
        neuron_type="ST-BIF",
        model_name="resnet", 
        is_softmax=False,  
        suppress_over_fire=False,
        record_inout=False,
        learnable=True,
        record_dir=output_dir
    )
    print("✅ 模型已转换为SNN")
    
    # 加载权重
    if not load_model_weights(snn_model, model_path, "SNN"):
        return None
    
    # 测试模型
    accuracy, inference_time = test_model(snn_model, test_loader, "SNN")
    return {"accuracy": accuracy, "inference_time": inference_time}


def compare_models(results):
    """比较模型性能"""
    print("\n" + "="*60)
    print("📊 模型性能对比")
    print("="*60)
    
    print(f"{'模型类型':<10} {'准确率':<10} {'推理时间':<15} {'相对速度':<10}")
    print("-" * 60)
    
    if 'ANN' in results and results['ANN'] is not None:
        ann_time = results['ANN']['inference_time']
        baseline_time = ann_time
    else:
        baseline_time = None
    
    for model_type, result in results.items():
        if result is not None:
            acc = result['accuracy']
            time_ms = result['inference_time'] * 1000
            
            if baseline_time:
                speed_ratio = baseline_time / result['inference_time']
                speed_str = f"{speed_ratio:.2f}x"
            else:
                speed_str = "N/A"
            
            print(f"{model_type:<10} {acc:<9.2f}% {time_ms:<13.3f}ms {speed_str:<10}")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='测试ANN/QANN/SNN模型性能')
    parser.add_argument('--models', nargs='+', choices=['ann', 'qann', 'snn', 'all'], 
                       default=['all'], help='要测试的模型类型')
    parser.add_argument('--batch-size', type=int, default=128, help='测试batch size')
    parser.add_argument('--ann-path', default='best_ANN.pth', help='ANN模型路径')
    parser.add_argument('--qann-path', default='best_QANN.pth', help='QANN模型路径') 
    parser.add_argument('--snn-path', default='best_SNN.pth', help='SNN模型路径')
    
    args = parser.parse_args()
    
    print("🎯 模型测试脚本")
    print(f"📱 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"📦 Batch Size: {args.batch_size}")
    
    # 准备数据
    print("\n📊 准备测试数据...")
    test_set = build_test_dataset()
    test_loader = create_test_dataloader(test_set, args.batch_size)
    print(f"✅ 测试集大小: {len(test_set)} 样本")
    
    # 测试模型
    results = {}
    
    if 'all' in args.models or 'ann' in args.models:
        results['ANN'] = test_ann_model(test_loader, args.ann_path)
    
    if 'all' in args.models or 'qann' in args.models:
        results['QANN'] = test_qann_model(test_loader, args.qann_path)
    
    if 'all' in args.models or 'snn' in args.models:
        results['SNN'] = test_snn_model(test_loader, args.snn_path)
    
    # 比较结果
    if len(results) > 1:
        compare_models(results)
    
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()