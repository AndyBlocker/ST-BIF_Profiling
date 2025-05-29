#!/usr/bin/env python3
"""
改进的转换测试脚本 - 使用更合理的量化参数
"""

import torch
import torch.nn as nn
import copy
from torchvision import datasets, transforms
from tqdm import tqdm
import time

# 导入必要的模块
from spike_quan_wrapper_ICML import SNNWrapper_MS, myquan_replace_resnet
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


def test_model(model, test_loader, model_name="Model", max_batches=None):
    """测试模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    print(f"🧪 测试{model_name}模型...")
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc=f"测试{model_name}", unit="batch")
        
        for batch_idx, (data, target) in enumerate(test_pbar):
            if max_batches and batch_idx >= max_batches:
                break
                
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 计算推理时间
            torch.cuda.synchronize()
            start_time = time.time()
            output = model(data)
            torch.cuda.synchronize()
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
    num_batches = max_batches if max_batches else len(test_loader)
    avg_time_per_batch = total_time / min(num_batches, len(test_loader))
    avg_time_per_sample = total_time / total
    
    print(f"✅ {model_name}测试完成:")
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


def test_different_quantization_levels(test_loader, ann_model_path="best_ANN.pth"):
    """测试不同量化级别的效果"""
    print("\n" + "="*70)
    print("🔬 改进的量化测试：不同Level对比")
    print("="*70)
    
    # 加载基础ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    if not load_model_weights(ann_model, ann_model_path, "ANN"):
        return None
    
    results = {}
    
    # 测试原始ANN
    print("\n🧠 测试原始ANN模型")
    ann_accuracy, ann_time = test_model(ann_model, test_loader, "ANN", max_batches=20)
    results['ANN'] = {"accuracy": ann_accuracy, "inference_time": ann_time}
    
    # 测试不同的量化级别
    levels = [4, 8, 16]  # 使用更温和的量化级别
    
    for level in levels:
        print(f"\n⚡ 测试Level {level}量化")
        
        # 深度复制模型
        qann_model = copy.deepcopy(ann_model)
        
        # 应用量化
        print(f"🔄 应用Level {level}量化...")
        myquan_replace_resnet(qann_model, level=level, weight_bit=32, is_softmax=False)
        
        # 测试量化模型
        qann_accuracy, qann_time = test_model(qann_model, test_loader, f"QANN-L{level}", max_batches=20)
        results[f'QANN-L{level}'] = {"accuracy": qann_accuracy, "inference_time": qann_time}
        
        # 清理内存
        del qann_model
        torch.cuda.empty_cache()
    
    return results


def test_optimized_conversion_pipeline(test_loader, ann_model_path="best_ANN.pth"):
    """
    优化的转换测试：ANN → QANN(合理level) → SNN
    """
    print("\n" + "="*70)
    print("🚀 优化的转换测试流水线: ANN → QANN(L32) → SNN")
    print("="*70)
    
    # 加载基础ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    if not load_model_weights(ann_model, ann_model_path, "ANN"):
        return None
    
    results = {}
    
    # 测试原始ANN
    print("\n🧠 步骤 1/3: 测试ANN模型")
    ann_accuracy, ann_time = test_model(ann_model, test_loader, "ANN", max_batches=30)
    results['ANN'] = {"accuracy": ann_accuracy, "inference_time": ann_time}
    
    # 转换为QANN (使用更合理的level=32)
    print("\n🔄 步骤 2/3: 转换为QANN (Level 32)")
    qann_model = copy.deepcopy(ann_model)
    myquan_replace_resnet(qann_model, level=16, weight_bit=32, is_softmax=False)
    
    print("\n⚡ 测试QANN模型")
    qann_accuracy, qann_time = test_model(qann_model, test_loader, "QANN", max_batches=30)
    results['QANN'] = {"accuracy": qann_accuracy, "inference_time": qann_time}
    
    # 转换为SNN
    print("\n🔄 步骤 3/3: 转换为SNN")
    import os
    output_dir = "/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/"
    os.makedirs(output_dir, exist_ok=True)
    
    snn_model = SNNWrapper_MS(
        ann_model=qann_model,
        cfg=None, 
        time_step=8,
        Encoding_type="analog", 
        level=16,  # 与QANN保持一致
        neuron_type="ST-BIF",
        model_name="resnet", 
        is_softmax=False,  
        suppress_over_fire=False,
        record_inout=False,
        learnable=True,
        record_dir=output_dir
    )
    
    print("\n🧠 测试SNN模型")
    snn_accuracy, snn_time = test_model(snn_model, test_loader, "SNN", max_batches=30)
    results['SNN'] = {"accuracy": snn_accuracy, "inference_time": snn_time}
    
    return results


def compare_results(results):
    """比较结果"""
    print("\n" + "="*80)
    print("📊 性能对比结果")
    print("="*80)
    
    print(f"{'模型类型':<15} {'准确率':<12} {'推理时间':<15} {'相对速度':<12} {'准确率变化':<12}")
    print("-" * 80)
    
    # 以ANN为基线
    baseline_acc = None
    baseline_time = None
    
    if 'ANN' in results and results['ANN'] is not None:
        baseline_acc = results['ANN']['accuracy']
        baseline_time = results['ANN']['inference_time']
    
    for model_type, result in results.items():
        if result is not None:
            acc = result['accuracy']
            time_ms = result['inference_time'] * 1000
            
            # 计算相对速度
            if baseline_time:
                speed_ratio = baseline_time / result['inference_time']
                speed_str = f"{speed_ratio:.2f}x"
            else:
                speed_str = "baseline"
            
            # 计算准确率变化
            if baseline_acc and model_type != 'ANN':
                acc_change = acc - baseline_acc
                acc_change_str = f"{acc_change:+.2f}%"
            else:
                acc_change_str = "baseline"
            
            print(f"{model_type:<15} {acc:<11.2f}% {time_ms:<13.3f}ms {speed_str:<12} {acc_change_str:<12}")
    
    print("-" * 80)


def main():
    print("🎯 改进的量化转换测试")
    print(f"📱 设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 准备数据
    print("\n📊 准备测试数据...")
    test_set = build_test_dataset()
    test_loader = create_test_dataloader(test_set, 128)
    print(f"✅ 测试集大小: {len(test_set)} 样本")
    
    # 测试1: 不同量化级别对比
    print("\n🔬 测试不同量化级别...")
    level_results = test_different_quantization_levels(test_loader)
    if level_results:
        compare_results(level_results)
    
    # 测试2: 优化的转换流水线
    print("\n🚀 测试优化的转换流水线...")
    pipeline_results = test_optimized_conversion_pipeline(test_loader)
    if pipeline_results:
        compare_results(pipeline_results)
    
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()