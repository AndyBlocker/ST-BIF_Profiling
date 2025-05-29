#!/usr/bin/env python3
"""
连续转换测试脚本 - 精简版本，用于性能分析
从一个训练好的ANN模型开始，连续转换对比：ANN → QANN → SNN
"""

import torch
import os
import argparse
import copy
from torchvision import datasets, transforms
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


def test_model(model, test_loader, model_name="Model", verbose=True):
    """测试模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_time = 0
    
    if verbose:
        print(f"Testing {model_name} model...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 计算推理时间
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if verbose and (batch_idx + 1) % 20 == 0:
                current_acc = 100. * correct / total
                print(f"  Batch {batch_idx+1:3d}/{len(test_loader)}: Acc={current_acc:.2f}%, Time={inference_time*1000:.1f}ms")
    
    accuracy = 100. * correct / total
    avg_time_per_batch = total_time / len(test_loader)
    avg_time_per_sample = total_time / total
    
    if verbose:
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Avg time per batch: {avg_time_per_batch*1000:.2f}ms")
        print(f"  Avg time per sample: {avg_time_per_sample*1000:.3f}ms")
        print(f"  Throughput: {1/avg_time_per_sample:.1f} samples/sec")
    
    return accuracy, avg_time_per_sample


def load_model_weights(model, model_path, model_name, verbose=True):
    """加载模型权重"""
    if not os.path.exists(model_path):
        if verbose:
            print(f"Model file not found: {model_path}")
        return False
    
    try:
        if verbose:
            print(f"Loading {model_name} model: {model_path}")
        
        # 尝试加载checkpoint格式
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if verbose and 'best_acc' in checkpoint:
                print(f"  Training best accuracy: {checkpoint['best_acc']:.2f}%")
        else:
            # 直接加载state_dict
            model.load_state_dict(checkpoint)
        
        if verbose:
            print(f"{model_name} model loaded successfully")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to load {model_name} model: {e}")
        return False


def test_conversion_pipeline(test_loader, ann_model_path="best_ANN.pth", qann_model_path="best_QANN.pth", verbose=True):
    """
    连续转换测试流水线：ANN → QANN → SNN
    测试训练好的ANN和QANN模型，以及从QANN转换的SNN
    """
    if verbose:
        print("\n" + "="*50)
        print("Conversion Pipeline: ANN → QANN → SNN")
        print("="*50)
    
    # 创建并加载ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    # 加载训练好的ANN权重
    if not load_model_weights(ann_model, ann_model_path, "ANN", verbose):
        if verbose:
            print("Failed to load base ANN model, stopping test")
        return None
    
    results = {}
    
    # 测试原始ANN模型
    if verbose:
        print("\nStep 1/3: Testing ANN model")
    ann_accuracy, ann_time = test_model(ann_model, test_loader, "ANN", verbose)
    results['ANN'] = {"accuracy": ann_accuracy, "inference_time": ann_time}
    
    # 加载训练好的QANN模型
    if verbose:
        print("\nStep 2/3: Loading trained QANN model")
    
    # 创建QANN模型结构
    qann_model = resnet.resnet18(pretrained=False)
    qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)
    
    # 尝试加载训练好的QANN模型
    if os.path.exists(qann_model_path):
        # 先应用量化结构，再加载权重
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
        if load_model_weights(qann_model, qann_model_path, "QANN", verbose):
            if verbose:
                print("Using trained QANN model")
        else:
            # 如果加载失败，从ANN转换
            if verbose:
                print("Failed to load trained QANN, converting from ANN")
            qann_model = copy.deepcopy(ann_model)
            myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    else:
        # 如果QANN文件不存在，从ANN转换
        if verbose:
            print("Trained QANN not found, converting from ANN")
        qann_model = copy.deepcopy(ann_model)
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    
    qann_accuracy, qann_time = test_model(qann_model, test_loader, "QANN", verbose)
    results['QANN'] = {"accuracy": qann_accuracy, "inference_time": qann_time}
    
    # 转换为SNN并测试
    if verbose:
        print("\nStep 3/3: Converting to SNN")
    
    # 确保输出目录存在
    output_dir = "/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用QANN模型创建SNN
    snn_model = SNNWrapper_MS(
        ann_model=qann_model,  # 使用训练好的QANN模型
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
    if verbose:
        print("Model converted to SNN")
    
    snn_accuracy, snn_time = test_model(snn_model, test_loader, "SNN", verbose)
    results['SNN'] = {"accuracy": snn_accuracy, "inference_time": snn_time}
    
    return results


def test_ann_to_qann_only(test_loader, ann_model_path="best_ANN.pth", qann_model_path="best_QANN.pth", verbose=True):
    """
    简化版本：只测试ANN和训练好的QANN
    """
    if verbose:
        print("\n" + "="*40)
        print("Simplified Test: ANN → QANN")
        print("="*40)
    
    # 加载基础ANN模型
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)
    
    if not load_model_weights(ann_model, ann_model_path, "ANN", verbose):
        return None
    
    results = {}
    
    # 测试ANN
    if verbose:
        print("\nTesting ANN model")
    ann_accuracy, ann_time = test_model(ann_model, test_loader, "ANN", verbose)
    results['ANN'] = {"accuracy": ann_accuracy, "inference_time": ann_time}
    
    # 加载训练好的QANN模型
    if verbose:
        print("\nLoading trained QANN model")
    
    # 创建QANN模型结构
    qann_model = resnet.resnet18(pretrained=False)
    qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)
    
    # 尝试加载训练好的QANN模型
    if os.path.exists(qann_model_path):
        # 先应用量化结构，再加载权重
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
        if load_model_weights(qann_model, qann_model_path, "QANN", verbose):
            if verbose:
                print("Using trained QANN model")
        else:
            # 如果加载失败，从ANN转换
            if verbose:
                print("Failed to load trained QANN, converting from ANN")
            qann_model = copy.deepcopy(ann_model)
            myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    else:
        # 如果QANN文件不存在，从ANN转换
        if verbose:
            print("Trained QANN not found, converting from ANN")
        qann_model = copy.deepcopy(ann_model)
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    
    qann_accuracy, qann_time = test_model(qann_model, test_loader, "QANN", verbose)
    results['QANN'] = {"accuracy": qann_accuracy, "inference_time": qann_time}
    
    return results


def compare_models(results):
    """比较模型性能"""
    print("\n" + "="*70)
    print("Performance Comparison")
    print("="*70)
    
    print(f"{'Model':<10} {'Accuracy':<12} {'Time (ms)':<12} {'Speed':<10} {'Acc Change':<12}")
    print("-" * 70)
    
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
            
            # 计算准确率损失
            if baseline_acc and model_type != 'ANN':
                acc_loss = baseline_acc - acc
                acc_loss_str = f"{acc_loss:+.2f}%"
            else:
                acc_loss_str = "baseline"
            
            print(f"{model_type:<10} {acc:<11.2f}% {time_ms:<11.3f} {speed_str:<10} {acc_loss_str:<12}")
    
    print("-" * 70)
    
    # 显示转换影响总结
    if len(results) >= 2:
        print("\nConversion Analysis:")
        
        if 'ANN' in results and 'QANN' in results:
            ann_acc = results['ANN']['accuracy']
            qann_acc = results['QANN']['accuracy']
            qann_speedup = results['ANN']['inference_time'] / results['QANN']['inference_time']
            print(f"  ANN → QANN: {ann_acc:.2f}% → {qann_acc:.2f}% ({qann_acc-ann_acc:+.2f}%), {qann_speedup:.2f}x speed")
        
        if 'QANN' in results and 'SNN' in results:
            qann_acc = results['QANN']['accuracy']
            snn_acc = results['SNN']['accuracy']
            snn_speedup = results['QANN']['inference_time'] / results['SNN']['inference_time']
            print(f"  QANN → SNN: {qann_acc:.2f}% → {snn_acc:.2f}% ({snn_acc-qann_acc:+.2f}%), {snn_speedup:.2f}x speed")
        
        if 'ANN' in results and 'SNN' in results:
            ann_acc = results['ANN']['accuracy']
            snn_acc = results['SNN']['accuracy']
            total_speedup = results['ANN']['inference_time'] / results['SNN']['inference_time']
            print(f"  Overall ANN → SNN: {ann_acc:.2f}% → {snn_acc:.2f}% ({snn_acc-ann_acc:+.2f}%), {total_speedup:.2f}x speed")


def main():
    parser = argparse.ArgumentParser(description='Conversion test script: ANN → QANN → SNN')
    parser.add_argument('--batch-size', type=int, default=128, help='Test batch size')
    parser.add_argument('--ann-path', default='best_ANN.pth', help='Base ANN model path')
    parser.add_argument('--qann-path', default='best_QANN.pth', help='Trained QANN model path')
    parser.add_argument('--skip-snn', action='store_true', help='Skip SNN test (ANN and QANN only)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output for profiling')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("Conversion Test Script")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"Batch Size: {args.batch_size}")
        print(f"ANN Model: {args.ann_path}")
        print(f"QANN Model: {args.qann_path}")
    
    # 准备数据
    if not args.quiet:
        print("\nPreparing test data...")
    test_set = build_test_dataset()
    test_loader = create_test_dataloader(test_set, args.batch_size)
    if not args.quiet:
        print(f"Test set size: {len(test_set)} samples")
    
    # 执行转换测试
    verbose = not args.quiet
    if args.skip_snn:
        if not args.quiet:
            print("\nSkipping SNN test mode")
        # 简化版本：只测试ANN和QANN
        results = test_ann_to_qann_only(test_loader, args.ann_path, args.qann_path, verbose)
    else:
        # 完整版本：测试ANN、QANN、SNN
        results = test_conversion_pipeline(test_loader, args.ann_path, args.qann_path, verbose)
    
    # 比较结果
    if results and len(results) > 1:
        compare_models(results)
    
    if not args.quiet:
        print("\nTest completed!")


if __name__ == "__main__":
    main()