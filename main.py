import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from spike_quan_wrapper_ICML import SNNWrapper_MS, myquan_replace_resnet
import resnet
from torchvision import datasets, transforms
import os
import time
import math
import multiprocessing as mp
import json
import glob
from datetime import datetime
from tqdm import tqdm

# build CIFAR10 dataset in /home/zilingwei/cifar10

def build_dataset():
    # CIFAR10的标准均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪+padding
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
    ])
    
    # 测试数据预处理（不使用增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    trainset = datasets.CIFAR10(root='/home/zilingwei/cifar10', train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root='/home/zilingwei/cifar10', train=False, download=True, transform=test_transform)
    return trainset, testset

def find_optimal_batch_size(model, device, dataset, min_batch_size=32, max_batch_size=2048):
    """
    自动寻找最优batch size - 通过实际训练步骤测试
    """
    print("🔍 自动寻找最优batch size...")
    
    if not torch.cuda.is_available():
        print(f"⚠️  未检测到CUDA，使用默认batch size: {min_batch_size}")
        return min_batch_size
    
    optimal_batch_size = min_batch_size
    model_clone = None
    
    # 创建一个模型副本用于测试
    try:
        import copy
        model_clone = copy.deepcopy(model)
        model_clone.to(device)
        model_clone.train()
    except:
        print("⚠️  无法创建模型副本，使用原模型测试")
        model_clone = model
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_size in [32, 64, 128, 256, 512, 768, 1024, 1536, 2048]:
        if batch_size > max_batch_size:
            break
            
        try:
            print(f"📊 测试batch size: {batch_size}")
            
            # 清空显存缓存
            torch.cuda.empty_cache()
            
            # 记录清空后的显存基线
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            
            # 创建临时数据加载器
            temp_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0  # 避免多进程干扰
            )
            
            # 创建优化器
            optimizer = optim.AdamW(model_clone.parameters(), lr=1e-4)
            
            # 执行几个完整的训练步骤来准确测试显存使用
            for i, (data, target) in enumerate(temp_loader):
                if i >= 3:  # 测试3个batch就够了
                    break
                    
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 前向传播
                output = model_clone(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 记录峰值显存使用
                current_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
            
            # 计算实际使用的显存
            actual_usage = (peak_memory - baseline_memory) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_ratio = peak_memory / (memory_total * 1024**3)  # 总显存比例
            
            print(f"   - 显存使用: {actual_usage:.2f}GB (峰值: {peak_memory/1024**3:.2f}GB/{memory_total:.2f}GB, {memory_ratio:.1%})")
            
            # 如果显存使用率超过85%，停止增加batch size
            if memory_ratio > 0.85:
                print(f"   - ⚠️  显存使用率过高，停止测试")
                break
            
            # 如果actual_usage过小，说明batch size还可以继续增加
            if actual_usage < 0.1:  # 小于100MB说明还有很大空间
                optimal_batch_size = batch_size
                continue
            
            optimal_batch_size = batch_size
            
            # 检查是否还有足够空间进一步增加
            if memory_ratio > 0.7:  # 超过70%就比较保守
                print(f"   - ✅ 显存使用适中，选择此batch size")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   - ❌ 显存不足，回退到上一个batch size")
                break
            else:
                print(f"   - ❌ 测试出错: {e}")
                raise e
        finally:
            torch.cuda.empty_cache()
    
    # 为了保险起见，最终选择的batch size稍微保守一点
    if optimal_batch_size >= 256:
        optimal_batch_size = int(optimal_batch_size * 0.8)  # 减少20%
        optimal_batch_size = max(optimal_batch_size, 128)  # 但不低于128
    
    print(f"🎯 最终选择batch size: {optimal_batch_size}")
    return optimal_batch_size

train_set, test_set = build_dataset()

# 首先创建模型来测试batch size
temp_model = resnet.resnet18(pretrained=True)
temp_model.fc = torch.nn.Linear(temp_model.fc.in_features, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_model.to(device)

# 自动确定最优batch size
batch_size = find_optimal_batch_size(temp_model, device, train_set)

# 清理临时模型
del temp_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def create_optimized_dataloader(dataset, batch_size, is_training=True, num_workers=None):
    """
    创建高度优化的DataLoader
    """
    # 自动计算最优worker数量
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        
        # 根据batch size和数据集大小调整worker数
        if batch_size <= 64:
            num_workers = min(4, cpu_count)
        elif batch_size <= 256:
            num_workers = min(6, cpu_count)
        elif batch_size <= 512:
            num_workers = min(8, cpu_count)
        else:  # batch_size > 512
            num_workers = min(4, cpu_count)  # 大batch size时减少worker避免内存竞争
    
    # 根据batch size调整prefetch factor
    if batch_size <= 128:
        prefetch_factor = 4  # 小batch需要更多预取
    elif batch_size <= 512:
        prefetch_factor = 3
    else:
        prefetch_factor = 2  # 大batch减少预取避免内存压力
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    
    # 配置参数
    loader_config = {
        'batch_size': batch_size,
        'shuffle': is_training,
        'num_workers': num_workers,
        'pin_memory': cuda_available,  # CUDA可用时启用
        'drop_last': is_training,  # 训练时丢弃不完整batch
        'persistent_workers': num_workers > 0,  # 持久化worker进程
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'generator': torch.Generator().manual_seed(42) if is_training else None,  # 固定随机种子
    }
    
    # 高级优化选项
    if cuda_available:
        # 使用non_blocking传输加速数据移动
        loader_config['pin_memory_device'] = 'cuda'
    
    # 对于大数据集，适当增加timeout
    if len(dataset) > 10000:
        loader_config['timeout'] = 60  # 60秒超时
    
    return torch.utils.data.DataLoader(dataset, **loader_config)

def setup_dataloader_optimization():
    """
    设置数据加载器的全局优化
    """
    # 设置多进程数据加载的共享策略
    if hasattr(torch.multiprocessing, 'set_sharing_strategy'):
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 优化OpenMP设置
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'
    
    # 优化MKL设置
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'
    
    print("⚙️  数据加载器全局优化已启用")

# 应用全局优化
setup_dataloader_optimization()

# 优化数据加载器配置
num_workers = min(8, os.cpu_count() or 4)
if batch_size >= 512:
    num_workers = min(num_workers, 4)

print(f"📦 数据加载器配置: batch_size={batch_size}, num_workers={num_workers}")
print(f"💻 CPU核心数: {os.cpu_count()}, CUDA可用: {torch.cuda.is_available()}")

# 创建优化的dataloaders
train_loader = create_optimized_dataloader(
    train_set, 
    batch_size=batch_size, 
    is_training=True,
    num_workers=num_workers
)

test_loader = create_optimized_dataloader(
    test_set, 
    batch_size=batch_size, 
    is_training=False,
    num_workers=num_workers
)

# 显示最终配置
print(f"🚀 训练数据加载器: batch_size={train_loader.batch_size}, num_workers={train_loader.num_workers}, prefetch_factor={train_loader.prefetch_factor}")
print(f"📋 测试数据加载器: batch_size={test_loader.batch_size}, num_workers={test_loader.num_workers}, prefetch_factor={test_loader.prefetch_factor}")
print(f"💾 高级选项: pin_memory={train_loader.pin_memory}, persistent_workers={train_loader.persistent_workers}")

# build model

model = resnet.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes

# train the ANN model and save it, validate its accuracy

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    带预热的余弦学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(state, checkpoint_dir, filename, is_best=False):
    """
    保存训练checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_' + filename)
        torch.save(state, best_filepath)
    
    print(f"💾 Checkpoint已保存: {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    加载checkpoint恢复训练状态
    """
    if not os.path.isfile(checkpoint_path):
        print(f"⚠️  Checkpoint文件不存在: {checkpoint_path}")
        return 0, 0.0, False
    
    print(f"📋 正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        return 0, 0.0, True  # 旧格式文件，只有模型状态
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 加载混合精度状态
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✅ 恢复训练: epoch={epoch}, best_acc={best_acc:.2f}%")
    return epoch, best_acc, True

def find_best_checkpoint(checkpoint_dir, model_name):
    """
    查找最佳的checkpoint文件（优先级：best > latest > 最新epoch）
    """
    if not os.path.exists(checkpoint_dir):
        return None, "best"
    
    # 第一优先级：查找 best 文件
    best_file = os.path.join(checkpoint_dir, f"best_{model_name}_checkpoint_*.pth")
    best_files = glob.glob(best_file)
    if best_files:
        # 如果有多个best文件，选择最新的
        latest_best = max(best_files, key=os.path.getmtime)
        return latest_best, "best"
    
    # 第二优先级：查找 best_{model_name}.pth 文件
    simple_best = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
    if os.path.exists(simple_best):
        return simple_best, "best"
    
    # 第三优先级：查找 latest 文件
    latest_file = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    if os.path.exists(latest_file):
        return latest_file, "latest"
    
    # 第四优先级：查找带epoch的checkpoint文件
    pattern = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if checkpoint_files:
        # 按修改时间排序，返回最新的
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint, "epoch"
    
    # 最后尝试：查找任何包含model_name的.pth文件
    pattern = os.path.join(checkpoint_dir, f"*{model_name}*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint, "fallback"
    
    return None, "none"

def create_checkpoint_dir(model_name):
    """
    创建 checkpoint 目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/{model_name}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def train_model(model, train_loader, test_loader, num_epochs=10, lr=5e-4, model_name="ANN", use_amp=True, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    
    # 使用AdamW优化器，weight_decay提升泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 计算总训练步数和预热步数
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10%的步数用于预热
    
    # 带预热的余弦学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # 混合精度训练
    scaler = GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    # 创建 checkpoint 目录
    checkpoint_dir = create_checkpoint_dir(model_name)
    
    # 初始化训练状态
    start_epoch = 0
    best_acc = 0.0
    
    # 尝试恢复训练
    if resume:
        print(f"🔍 查找最佳checkpoint文件: {model_name}")
        
        # 优先查找最佳模型
        checkpoint_path, checkpoint_type = find_best_checkpoint("checkpoints", model_name)
        
        if checkpoint_path:
            type_emoji = {
                "best": "🏆",
                "latest": "🔄", 
                "epoch": "📅",
                "fallback": "🔍"
            }
            print(f"{type_emoji.get(checkpoint_type, '📋')} 找到{checkpoint_type}checkpoint: {checkpoint_path}")
            
            start_epoch, best_acc, loaded = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler
            )
            if loaded:
                if checkpoint_type == "best":
                    # 从最佳模型开始，重新训练以进一步优化
                    print(f"🏆 从最佳模型开始（best_acc={best_acc:.2f}%），继续优化")
                    # 可以选择从最佳epoch继续，或者重新开始但保持best_acc
                    start_epoch = max(0, start_epoch)  # 从最佳epoch继续
                else:
                    start_epoch += 1  # 从下一个epoch开始
                    print(f"🔄 恢复训练，从 epoch {start_epoch} 开始")
        else:
            print(f"🎆 未找到任何checkpoint，从头开始训练")
            print(f"    - 查找路径: checkpoints/")
            print(f"    - checkpoints目录存在: {os.path.exists('checkpoints')}")
    
    print(f"🚀 开始训练{model_name}模型...")
    print(f"📊 设备: {device}, 混合精度: {use_amp and torch.cuda.is_available()}, Batch Size: {train_loader.batch_size}")
    print(f"📈 总步数: {num_training_steps}, 预热步数: {num_warmup_steps}")
    print(f"💾 Checkpoint目录: {checkpoint_dir}")
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f"训练{model_name}", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # 使用tqdm显示batch进度
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, (data, target) in enumerate(batch_pbar):
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
            
            scheduler.step()  # 每个batch后更新学习率
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新batch进度条
            current_acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total
        test_acc = validate_model(model, test_loader)
        
        # 检查是否是最佳模型
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        # 保存checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss': running_loss / len(train_loader),
        }
        
        if scaler is not None:
            checkpoint_state['scaler_state_dict'] = scaler.state_dict()
        
        # 每10个epoch或最后一个epoch保存checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1 or is_best:
            save_checkpoint(
                checkpoint_state, 
                checkpoint_dir, 
                f"{model_name}_checkpoint_epoch_{epoch:03d}.pth",
                is_best=is_best
            )
        
        # 如果是最佳模型，也保存一个简单名称的best文件
        if is_best:
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"🏆 最佳模型已更新: best_{model_name}.pth (acc={best_acc:.2f}%)")
        
        # 保存最新的checkpoint（用于resume）
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(
            checkpoint_state,
            "checkpoints", 
            f"{model_name}_latest.pth"
        )
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%',
            'Best': f'{best_acc:.2f}%',
            'Time': f'{epoch_time:.1f}s'
        })
    
    print(f"✅ {model_name}训练完成，最佳测试准确率: {best_acc:.2f}%")
    return model

def validate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        # 使用tqdm显示验证进度
        val_pbar = tqdm(test_loader, desc="验证中", leave=False, unit="batch")
        
        for data, target in val_pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 使用混合精度推理
            if torch.cuda.is_available():
                with autocast('cuda'):
                    output = model(data)
            else:
                output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新验证进度条
            current_acc = 100. * correct / total
            val_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    accuracy = 100. * correct / total
    return accuracy

# 训练ANN模型
print("=== 第一阶段：训练ANN模型 ===")
# 长时间训练以获得最佳性能，自动从最佳checkpoint恢复
model = train_model(model, train_loader, test_loader, num_epochs=200, lr=0.01, model_name="ANN", resume=True)
torch.save(model.state_dict(), 'ANN.pth')
print("ANN模型已保存为 ANN.pth")

# 第二阶段：训练独立的QANN模型
print("\n=== 第二阶段：训练QANN模型 ===")

# 创建新的QANN模型（从ANN模型初始化）
print("创建QANN模型...")
qann_model = resnet.resnet18(pretrained=False)
qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)

# 加载训练好的ANN权重作为QANN的初始化
print("使用训练好的ANN权重初始化QANN模型...")
qann_model.load_state_dict(model.state_dict())

# 转换为量化模型
print("应用量化操作...")
myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
print("模型已转换为QANN结构")

# 验证转换后的QANN模型准确率
qann_acc_initial = validate_model(qann_model, test_loader)
print(f"初始QANN模型验证准确率: {qann_acc_initial:.2f}%")

# 训练QANN模型
print("\n开始训练QANN模型...")
# QANN需要重新训练以适应量化操作，从头训练获得最佳量化性能
qann_model = train_model(qann_model, train_loader, test_loader, num_epochs=100, lr=0.001, model_name="QANN", resume=True)

# 保存训练好的QANN模型
torch.save(qann_model.state_dict(), 'QANN.pth')
# 同时保存最佳QANN模型
torch.save(qann_model.state_dict(), 'best_QANN.pth')
print("QANN模型已保存为 QANN.pth 和 best_QANN.pth")

# convert QANN model to SNN model, validate its accuracy
print("\n=== 第三阶段：转换为SNN模型 ===")

# 确保输出目录存在
output_dir = "/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/"
os.makedirs(output_dir, exist_ok=True)

# 使用训练好的QANN模型创建SNN
print("使用训练好的QANN模型创建SNN...")
snn_model = SNNWrapper_MS(ann_model=qann_model, cfg=None, time_step=8, \
                    Encoding_type="analog", level=8, neuron_type="ST-BIF", \
                    model_name="resnet", is_softmax = False,  suppress_over_fire = False,\
                    record_inout=False,learnable=True,record_dir=output_dir)
print("模型已转换为SNN")

def train_snn_model(snn_model, train_loader, test_loader, num_epochs=100, lr=0.001, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snn_model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # 轻微标签平滑
    optimizer = optim.AdamW(snn_model.parameters(), lr=lr, weight_decay=5e-5)
    
    # 计算总训练步数和预热步数
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)  # 5%的步数用于预热（SNN较敏感）
    
    # 带预热的余弦学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # 创建 checkpoint 目录
    checkpoint_dir = create_checkpoint_dir("SNN")
    
    # 初始化训练状态
    start_epoch = 0
    best_acc = 0.0
    
    # 尝试恢复训练
    if resume:
        print(f"🔍 查找最佳SNN checkpoint文件")
        
        checkpoint_path, checkpoint_type = find_best_checkpoint("checkpoints", "SNN")
        
        if checkpoint_path:
            type_emoji = {
                "best": "🏆",
                "latest": "🔄", 
                "epoch": "📅",
                "fallback": "🔍"
            }
            print(f"{type_emoji.get(checkpoint_type, '📋')} 找到SNN {checkpoint_type}checkpoint: {checkpoint_path}")
            
            start_epoch, best_acc, loaded = load_checkpoint(
                checkpoint_path, snn_model, optimizer, scheduler
            )
            if loaded:
                if checkpoint_type == "best":
                    print(f"🏆 从最佳SNN模型开始（best_acc={best_acc:.2f}%），继续优化")
                    start_epoch = max(0, start_epoch)
                else:
                    start_epoch += 1
                    print(f"🔄 恢复SNN训练，从 epoch {start_epoch} 开始")
        else:
            print(f"🎆 未找到SNN checkpoint，从头开始训练")
            print(f"    - 查找路径: checkpoints/")
            print(f"    - checkpoints目录存在: {os.path.exists('checkpoints')}")
    
    print(f"🧠 开始训练SNN模型...")
    print(f"📊 总步数: {num_training_steps}, 预热步数: {num_warmup_steps}")
    print(f"💾 Checkpoint目录: {checkpoint_dir}")
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="训练SNN", unit="epoch")
    
    for epoch in epoch_pbar:
        snn_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # 使用tqdm显示batch进度
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, (data, target) in enumerate(batch_pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # SNN模型通常不适合混合精度，因为涉及复杂的脉冲操作
            output = snn_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(snn_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # 每个batch后更新学习率
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新batch进度条
            current_acc = 100. * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        epoch_time = time.time() - start_time
        train_acc = 100. * correct / total
        test_acc = validate_snn_model(snn_model, test_loader)
        
        # 检查是否是最佳模型
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        
        # 保存checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': snn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss': running_loss / len(train_loader),
        }
        
        # 每15个epoch或最后一个epoch保存checkpoint
        if (epoch + 1) % 15 == 0 or epoch == num_epochs - 1 or is_best:
            save_checkpoint(
                checkpoint_state, 
                checkpoint_dir, 
                f"SNN_checkpoint_epoch_{epoch:03d}.pth",
                is_best=is_best
            )
        
        # 如果是最佳模型，也保存一个简单名称的best文件
        if is_best:
            torch.save(snn_model.state_dict(), 'best_SNN.pth')
            print(f"🏆 最佳SNN模型已更新: best_SNN.pth (acc={best_acc:.2f}%)")
        
        # 保存最新的checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(
            checkpoint_state,
            "checkpoints", 
            "SNN_latest.pth"
        )
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%',
            'Best': f'{best_acc:.2f}%',
            'Time': f'{epoch_time:.1f}s'
        })
    
    print(f"✅ SNN训练完成，最佳测试准确率: {best_acc:.2f}%")
    return snn_model

def validate_snn_model(snn_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snn_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = snn_model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'SNN验证准确率: {accuracy:.2f}%')
    return accuracy

# 验证初始SNN模型准确率
snn_acc_initial = validate_snn_model(snn_model, test_loader)
print(f"初始SNN模型验证准确率: {snn_acc_initial:.2f}%")

# train the SNN model and save it as SNN.pth
print("\n开始训练SNN模型...")
# SNN长时间训练以优化时序特性，自动从最佳checkpoint恢复
snn_model = train_snn_model(snn_model, train_loader, test_loader, num_epochs=100, lr=0.001, resume=True)
torch.save(snn_model.state_dict(), 'SNN.pth')
print("SNN模型已保存为 SNN.pth")

print("\n=== 训练完成！模型性能总结 ===")
print(f"最终SNN模型验证准确率: {validate_snn_model(snn_model, test_loader):.2f}%")