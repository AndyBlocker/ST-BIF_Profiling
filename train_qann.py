#!/usr/bin/env python3
"""
快速训练QANN模型的脚本
从训练好的ANN模型开始，创建并训练一个量化的QANN模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import resnet
from spike_quan_wrapper_ICML import myquan_replace_resnet
from torchvision import datasets, transforms
import os
import time
from tqdm import tqdm

def build_dataset():
    """构建CIFAR10数据集"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 测试数据预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    trainset = datasets.CIFAR10(root='/home/zilingwei/cifar10', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='/home/zilingwei/cifar10', train=False, download=True, transform=test_transform)
    return trainset, testset

def validate_model(model, test_loader):
    """验证模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def train_qann_model(model, train_loader, test_loader, num_epochs=50, lr=0.001):
    """训练QANN模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 混合精度训练
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"Starting QANN training for {num_epochs} epochs...")
    print(f"Device: {device}, Learning rate: {lr}")
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # 训练循环
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for data, target in train_pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            current_acc = 100. * correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        
        # 验证
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total
        test_acc = validate_model(model, test_loader)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, "
              f"Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_QANN.pth')
            print(f"  --> New best model saved! Best accuracy: {best_acc:.2f}%")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'QANN.pth')
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    print("Models saved as 'best_QANN.pth' and 'QANN.pth'")
    
    return model

def main():
    print("QANN Training Script")
    print("=" * 40)
    
    # 构建数据集
    print("Loading CIFAR10 dataset...")
    train_set, test_set = build_dataset()
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    print(f"Batch size: {batch_size}")
    
    # 创建QANN模型
    print("\nCreating QANN model...")
    qann_model = resnet.resnet18(pretrained=False)
    qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)
    
    # 尝试从训练好的ANN模型初始化
    ann_path = 'best_ANN.pth'
    if os.path.exists(ann_path):
        print(f"Loading ANN weights from {ann_path}...")
        try:
            checkpoint = torch.load(ann_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                qann_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                qann_model.load_state_dict(checkpoint)
            print("ANN weights loaded successfully")
        except Exception as e:
            print(f"Failed to load ANN weights: {e}")
            print("Starting with random initialization")
    else:
        print(f"ANN model not found at {ann_path}, using random initialization")
    
    # 应用量化
    print("Applying quantization...")
    myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    print("Model converted to QANN structure")
    
    # 验证初始性能
    initial_acc = validate_model(qann_model, test_loader)
    print(f"Initial QANN accuracy: {initial_acc:.2f}%")
    
    # 训练QANN模型
    print("\nStarting QANN training...")
    trained_model = train_qann_model(qann_model, train_loader, test_loader, num_epochs=50, lr=0.001)
    
    # 最终验证
    final_acc = validate_model(trained_model, test_loader)
    print(f"\nFinal QANN accuracy: {final_acc:.2f}%")
    print(f"Improvement: {final_acc - initial_acc:+.2f}%")

if __name__ == "__main__":
    main()