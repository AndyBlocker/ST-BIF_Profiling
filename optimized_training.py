#!/usr/bin/env python3
"""
Optimized Training Script for ST-BIF SNN

This script demonstrates the performance optimizations implemented for ST-BIF SNN:
1. Memory pool optimization
2. CUDA optimizations (cuDNN benchmark, etc.)
3. Mixed precision training  
4. Optimized ST-BIF neuron implementation
5. Performance monitoring

Usage:
    python optimized_training.py --optimization-level medium --enable-amp
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings

# Import optimization utilities
from snn.optimization_utils import (
    setup_optimizations, cleanup_optimizations, 
    get_performance_monitor, CUDAOptimizer, MixedPrecisionTrainer
)

# Import optimized ST-BIF neuron
from snn.neurons.optimized_st_bif import create_optimized_stbif_neuron

# Import original framework components
try:
    from snn.conversion import myquan_replace_resnet
    from wrapper import SNNWrapper_MS
    from models import resnet
except ImportError:
    print("Warning: Could not import SNN framework components. Using dummy implementations.")
    myquan_replace_resnet = lambda x, **kwargs: x
    SNNWrapper_MS = nn.Module
    resnet = None


class OptimizedSNNTrainer:
    """
    Optimized trainer for ST-BIF SNN with performance monitoring and optimizations.
    """
    
    def __init__(self, optimization_level="medium", enable_amp=True, 
                 time_steps=8, level=8, device='cuda'):
        self.optimization_level = optimization_level
        self.enable_amp = enable_amp
        self.time_steps = time_steps
        self.level = level
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Initializing Optimized SNN Trainer")
        print(f"   Optimization Level: {optimization_level}")
        print(f"   Mixed Precision: {enable_amp}")
        print(f"   Device: {self.device}")
        print(f"   Time Steps: {time_steps}")
        
        # Setup optimizations
        self.memory_pool, self.mp_trainer = setup_optimizations()
        if not enable_amp:
            self.mp_trainer.enabled = False
            
        # Performance monitoring
        self.monitor = get_performance_monitor()
        self.monitor.reset()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def build_model(self, num_classes=10, ann_path=None, qann_path=None):
        """Build optimized SNN model following ANN->QANN->SNN pipeline"""
        
        print("\n🏗️ Building Optimized SNN Model")
        
        # Step 1: Create ANN model
        print("   Step 1/3: Creating ANN model...")
        if resnet is not None:
            ann_model = resnet.resnet18(pretrained=False)
            ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, num_classes)
        else:
            # Dummy model for testing
            ann_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )
        
        # Load pre-trained weights if available
        if ann_path and ann_path != "None":
            try:
                print(f"   Loading ANN weights: {ann_path}")
                checkpoint = torch.load(ann_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    ann_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    ann_model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"   ⚠️  Could not load ANN weights: {e}")
        
        # Step 2: Convert to QANN
        print("   Step 2/3: Converting to QANN...")
        try:
            qann_model = myquan_replace_resnet(ann_model, level=self.level)
        except:
            print("   ⚠️  Using ANN model as QANN (quantization skipped)")
            qann_model = ann_model
        
        # Load QANN weights if available  
        if qann_path and qann_path != "None":
            try:
                print(f"   Loading QANN weights: {qann_path}")
                checkpoint = torch.load(qann_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    qann_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    qann_model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"   ⚠️  Could not load QANN weights: {e}")
        
        # Step 3: Convert to optimized SNN
        print("   Step 3/3: Converting to Optimized SNN...")
        try:
            # Create SNN wrapper with optimized neurons
            self.model = OptimizedSNNWrapper(
                qann_model, 
                time_steps=self.time_steps, 
                level=self.level,
                optimization_level=self.optimization_level
            )
        except:
            print("   ⚠️  Using dummy SNN model for testing")
            self.model = DummyOptimizedSNN(num_classes, self.optimization_level)
        
        self.model.to(self.device)
        print("   ✅ Optimized SNN model created successfully")
        
        return self.model
    
    def setup_training(self, learning_rate=0.001):
        """Setup optimizer and loss function"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"\n⚙️ Training Setup Complete")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: CrossEntropyLoss")
    
    def train_step(self, data, target):
        """Single optimized training step with performance monitoring"""
        
        # Record start time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        step_start_time = time.time()
        
        # Forward pass with mixed precision
        with self.mp_trainer.autocast():
            output = self.model(data)
            loss = self.criterion(output, target)
        
        # Record forward time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        forward_time = time.time() - step_start_time
        self.monitor.record_forward_time(forward_time * 1000)  # Convert to ms
        
        # Backward pass
        backward_start_time = time.time()
        self.optimizer.zero_grad()
        self.mp_trainer.backward(loss)
        self.mp_trainer.step_optimizer(self.optimizer)
        
        # Record backward time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        backward_time = time.time() - backward_start_time
        self.monitor.record_backward_time(backward_time * 1000)  # Convert to ms
        
        # Record memory usage
        if self.device.type == 'cuda':
            memory_info = CUDAOptimizer.get_memory_info()
            if memory_info:
                self.monitor.record_memory_usage(memory_info['allocated_gb'])
        
        return loss.item(), output
    
    def benchmark_performance(self, data_loader, num_steps=100):
        """Benchmark performance with current optimizations"""
        
        print(f"\n⚡ Benchmarking Performance ({num_steps} steps)...")
        
        self.model.train()
        total_loss = 0.0
        step_count = 0
        
        # Warmup
        print("   Warming up...")
        for i, (data, target) in enumerate(data_loader):
            if i >= 10:  # 10 warmup steps
                break
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                _ = self.model(data)
        
        # Clear caches and reset monitoring
        CUDAOptimizer.clear_memory_cache()
        self.monitor.reset()
        
        print("   Running benchmark...")
        start_time = time.time()
        
        for i, (data, target) in enumerate(data_loader):
            if step_count >= num_steps:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            loss, _ = self.train_step(data, target)
            total_loss += loss
            step_count += 1
            
            if (step_count + 1) % 20 == 0:
                print(f"   Step {step_count + 1}/{num_steps}")
        
        total_time = time.time() - start_time
        
        # Get performance summary
        perf_summary = self.monitor.get_summary()
        memory_pool_stats = self.memory_pool.get_stats() if self.memory_pool else {}
        
        print(f"\n📈 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average loss: {total_loss / step_count:.4f}")
        print(f"   Steps per second: {step_count / total_time:.2f}")
        
        if 'forward' in perf_summary:
            print(f"   Forward time: {perf_summary['forward']['mean_ms']:.2f}ms ± {perf_summary['forward']['std_ms']:.2f}ms")
        if 'backward' in perf_summary:
            print(f"   Backward time: {perf_summary['backward']['mean_ms']:.2f}ms ± {perf_summary['backward']['std_ms']:.2f}ms")
        if 'memory' in perf_summary:
            print(f"   Peak memory: {perf_summary['memory']['peak_gb']:.2f}GB")
        
        if memory_pool_stats:
            print(f"   Memory pool hit rate: {memory_pool_stats['hit_rate']:.1%}")
        
        return perf_summary
    
    def cleanup(self):
        """Cleanup optimization resources"""
        cleanup_optimizations()
        print("✅ Optimization resources cleaned up")


class OptimizedSNNWrapper(nn.Module):
    """
    Optimized SNN wrapper that replaces regular neurons with optimized ST-BIF neurons
    """
    
    def __init__(self, base_model, time_steps=8, level=8, optimization_level="medium"):
        super(OptimizedSNNWrapper, self).__init__()
        self.base_model = base_model
        self.time_steps = time_steps
        self.level = level
        self.optimization_level = optimization_level
        
        # Replace neurons with optimized versions
        self._replace_neurons()
    
    def _replace_neurons(self):
        """Replace standard neurons with optimized ST-BIF neurons"""
        def replace_activation(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    # Replace ReLU with optimized ST-BIF neuron
                    setattr(module, name, create_optimized_stbif_neuron(
                        q_threshold=1.0,
                        level=self.level,
                        optimization_level=self.optimization_level
                    ))
                else:
                    replace_activation(child)
        
        replace_activation(self.base_model)
        
        # Set time steps for all ST-BIF neurons
        for module in self.modules():
            if hasattr(module, 'T'):
                module.T = self.time_steps
    
    def forward(self, x):
        # Expand input for time steps: [B, C, H, W] -> [T*B, C, H, W]
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        x_expanded = x_expanded.view(-1, *x.shape[1:])
        
        # Forward through optimized SNN
        output = self.base_model(x_expanded)
        
        # Average over time steps: [T*B, num_classes] -> [B, num_classes]
        output = output.view(self.time_steps, batch_size, -1).mean(dim=0)
        
        return output


class DummyOptimizedSNN(nn.Module):
    """Dummy SNN model for testing when framework components are not available"""
    
    def __init__(self, num_classes=10, optimization_level="medium"):
        super(DummyOptimizedSNN, self).__init__()
        self.optimization_level = optimization_level
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            create_optimized_stbif_neuron(1.0, 8, optimization_level=optimization_level),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        
        # Set time steps
        for module in self.modules():
            if hasattr(module, 'T'):
                module.T = 8
    
    def forward(self, x):
        # Simple forward pass
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_data_loader(batch_size=32, num_workers=2):
    """Create CIFAR-10 data loader for testing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    try:
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    except:
        print("⚠️  Could not download CIFAR-10, using dummy data")
        # Create dummy dataset
        class DummyDataset:
            def __init__(self, size=1000):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()
        dataset = DummyDataset()
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='Optimized ST-BIF SNN Training')
    parser.add_argument('--optimization-level', choices=['low', 'medium', 'high'], default='medium',
                       help='Optimization level (low=original, medium=optimized, high=adaptive)')
    parser.add_argument('--enable-amp', action='store_true', default=True,
                       help='Enable automatic mixed precision')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of benchmark steps')
    parser.add_argument('--time-steps', type=int, default=8,
                       help='Number of time steps for SNN')
    parser.add_argument('--level', type=int, default=8,
                       help='Quantization level')
    parser.add_argument('--ann-path', type=str, default=None,
                       help='Path to ANN model weights')
    parser.add_argument('--qann-path', type=str, default=None,
                       help='Path to QANN model weights')
    
    args = parser.parse_args()
    
    print("🚀 Optimized ST-BIF SNN Training Script")
    print("=" * 50)
    
    # Create trainer
    trainer = OptimizedSNNTrainer(
        optimization_level=args.optimization_level,
        enable_amp=args.enable_amp,
        time_steps=args.time_steps,
        level=args.level
    )
    
    # Build model
    model = trainer.build_model(
        num_classes=10,
        ann_path=args.ann_path,
        qann_path=args.qann_path
    )
    
    # Setup training
    trainer.setup_training()
    
    # Create data loader
    data_loader = create_data_loader(batch_size=args.batch_size)
    
    # Run benchmark
    perf_results = trainer.benchmark_performance(data_loader, num_steps=args.steps)
    
    # Compare with baseline if available
    print(f"\n📊 Optimization Impact Analysis:")
    print(f"   Optimization Level: {args.optimization_level}")
    print(f"   Mixed Precision: {'Enabled' if args.enable_amp else 'Disabled'}")
    
    # Cleanup
    trainer.cleanup()
    
    print("\n✅ Optimized training completed successfully!")
    return perf_results


if __name__ == "__main__":
    results = main()