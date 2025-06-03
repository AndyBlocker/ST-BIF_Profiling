#!/usr/bin/env python3
"""
ANN to SNN Conversion Example

This example demonstrates the complete ANN → QANN → SNN conversion pipeline
using the modular ST-BIF framework. It showcases:

1. Loading a standard ANN model (ResNet18)
2. Converting to quantized ANN (QANN) with learnable quantizers
3. Converting to spiking neural network (SNN) with ST-BIF neurons
4. Performance comparison across all three model types

Example usage:
    python examples/ann_to_snn_conversion.py
    python examples/ann_to_snn_conversion.py --batch-size 64 --quiet
"""

import sys
import os
import argparse
import copy
import time
import torch
from torchvision import datasets, transforms


# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular SNN framework components
from snn.conversion import myquan_replace_resnet
from snn.optimization_utils import setup_optimizations
from snn.optimization_utils_fixed import setup_st_bif_optimizations
from wrapper import SNNWrapper_MS
from models import resnet

# Import for backward compatibility verification
try:
    from snn import ST_BIFNeuron_MS, MyQuan, LLConv2d_MS, LLLinear_MS
    BACKWARD_COMPAT = True
except ImportError as e:
    print(f"⚠️  Some components not available: {e}")
    BACKWARD_COMPAT = False


def build_test_dataset():
    """Build CIFAR-10 test dataset with standard preprocessing"""
    # CIFAR-10 normalization constants
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Test data preprocessing (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    testset = datasets.CIFAR10(
        root='/home/zilingwei/cifar10', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    return testset


def create_test_dataloader(dataset, batch_size=128):
    """Create test data loader with optimal settings"""
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    return test_loader


def test_model(model, test_loader, model_name="Model", verbose=True):
    """Test model accuracy and performance"""
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
            
            # Measure inference time
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
    """Load model weights from checkpoint"""
    if not os.path.exists(model_path):
        if verbose:
            print(f"Model file not found: {model_path}")
        return False
    
    try:
        if verbose:
            print(f"Loading {model_name} model: {model_path}")
        
        # Try loading checkpoint format
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if verbose and 'best_acc' in checkpoint:
                print(f"  Training best accuracy: {checkpoint['best_acc']:.2f}%")
        else:
            # Direct state_dict loading
            model.load_state_dict(checkpoint)
        
        if verbose:
            print(f"{model_name} model loaded successfully")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to load {model_name} model: {e}")
        return False


def run_conversion_pipeline(test_loader, ann_model_path="checkpoints/resnet/best_ANN.pth", 
                          qann_model_path="checkpoints/resnet/best_QANN.pth", verbose=True):
    """
    Execute the complete ANN → QANN → SNN conversion pipeline
    
    Args:
        test_loader: PyTorch DataLoader for testing
        ann_model_path: Path to trained ANN model weights  
        qann_model_path: Path to trained QANN model weights
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary containing results for each model type
    """
    if verbose:
        print("\n" + "="*60)
        print("🧪 ANN TO SNN CONVERSION PIPELINE")
        print("Testing: ANN → QANN → SNN using modular framework")
        print("="*60)
    
    results = {}
    
    # Step 1: Load and test ANN model
    if verbose:
        print("\n🔹 Step 1/3: Loading and testing ANN model")
    
    ann_model = resnet.resnet18(pretrained=False)
    ann_model.fc = torch.nn.Linear(ann_model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    
    if not load_model_weights(ann_model, ann_model_path, "ANN", verbose):
        if verbose:
            print("❌ Failed to load base ANN model, stopping pipeline")
        return None
    
    ann_accuracy, ann_time = test_model(ann_model, test_loader, "ANN", verbose)
    results['ANN'] = {"accuracy": ann_accuracy, "inference_time": ann_time}
    
    # Step 2: Convert to QANN and test
    if verbose:
        print("\n🔹 Step 2/3: Converting ANN to QANN")
    
    qann_model = resnet.resnet18(pretrained=False)
    qann_model.fc = torch.nn.Linear(qann_model.fc.in_features, 10)
    
    # Try loading pre-trained QANN model
    if os.path.exists(qann_model_path):
        if verbose:
            print("  🔧 Applying quantization structure...")
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
        
        if load_model_weights(qann_model, qann_model_path, "QANN", verbose):
            if verbose:
                print("  ✓ Using pre-trained QANN weights")
        else:
            if verbose:
                print("  ⚠️  Failed to load pre-trained QANN, converting from ANN")
            qann_model = copy.deepcopy(ann_model)
            myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    else:
        if verbose:
            print("  ⚠️  Pre-trained QANN not found, converting from ANN")
        qann_model = copy.deepcopy(ann_model)
        myquan_replace_resnet(qann_model, level=8, weight_bit=32, is_softmax=False)
    
    qann_accuracy, qann_time = test_model(qann_model, test_loader, "QANN", verbose)
    results['QANN'] = {"accuracy": qann_accuracy, "inference_time": qann_time}
    
    # Step 3: Convert to SNN and test
    if verbose:
        print("\n🔹 Step 3/3: Converting QANN to SNN")
    
    # Ensure output directory exists
    output_dir = "/home/zilingwei/output_bin_snn_resnet_w32_a4_T8/"
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("  🔧 Creating SNN wrapper with ST-BIF neurons...")
    
    # setup_optimizations()
#     optimizer = setup_st_bif_optimizations(
#       enable_memory_pool=True,
#       enable_tf32=True,
#       enable_cudnn_benchmark=False  # 对变长输入更安全
#   )
    # Create SNN using the modular SNNWrapper_MS
    snn_model = SNNWrapper_MS(
        ann_model=qann_model,  # Use the quantized model as base
        cfg=None, 
        time_step=8,           # 8 time steps for temporal encoding
        Encoding_type="analog", 
        level=8,               # Quantization level
        neuron_type="ST-BIF",  # Use ST-BIF (Spike Threshold Bifurcation) neurons
        model_name="resnet", 
        is_softmax=False,  
        suppress_over_fire=False,
        record_inout=False,    # Disable I/O recording for faster inference
        learnable=True,
        record_dir=output_dir
    )
    
    if verbose:
        print("  ✓ SNN conversion completed")
    
    snn_accuracy, snn_time = test_model(snn_model, test_loader, "SNN", verbose)
    results['SNN'] = {"accuracy": snn_accuracy, "inference_time": snn_time}
    
    return results


def display_results(results):
    """Display comprehensive results comparison"""
    print("\n" + "="*70)
    print("📊 CONVERSION PIPELINE RESULTS")
    print("="*70)
    
    print(f"{'Model':<10} {'Accuracy':<12} {'Time (ms)':<12} {'Speed':<10} {'Acc Change':<12}")
    print("-" * 70)
    
    # Use ANN as baseline
    baseline_acc = results.get('ANN', {}).get('accuracy')
    baseline_time = results.get('ANN', {}).get('inference_time')
    
    for model_type, result in results.items():
        if result is not None:
            acc = result['accuracy']
            time_ms = result['inference_time'] * 1000
            
            # Calculate relative speed
            if baseline_time and model_type != 'ANN':
                speed_ratio = baseline_time / result['inference_time']
                speed_str = f"{speed_ratio:.2f}x"
            else:
                speed_str = "baseline"
            
            # Calculate accuracy change
            if baseline_acc and model_type != 'ANN':
                acc_change = acc - baseline_acc
                acc_change_str = f"{acc_change:+.2f}%"
            else:
                acc_change_str = "baseline"
            
            print(f"{model_type:<10} {acc:.2f}%{'':<4} {time_ms:<11.3f} {speed_str:<10} {acc_change_str:<12}")
    
    print("-" * 70)
    
    # Display conversion analysis
    if len(results) >= 2:
        print("\n🔍 Conversion Analysis:")
        
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
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='ANN to SNN Conversion Example using ST-BIF Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/ann_to_snn_conversion.py
  python examples/ann_to_snn_conversion.py --batch-size 64
  python examples/ann_to_snn_conversion.py --quiet
        """
    )
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Test batch size (default: 128)')
    parser.add_argument('--ann-path', default='checkpoints/resnet/best_ANN.pth', 
                       help='Path to ANN model weights')
    parser.add_argument('--qann-path', default='checkpoints/resnet/best_QANN.pth', 
                       help='Path to QANN model weights')
    parser.add_argument('--quiet', action='store_true', 
                       help='Minimal output for benchmarking')
    
    args = parser.parse_args()
    
    # Display header information
    if not args.quiet:
        print("🧪 ANN TO SNN CONVERSION EXAMPLE")
        print("="*50)
        print(f"Framework: ST-BIF Modular Framework")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"Batch Size: {args.batch_size}")
        print(f"ANN Model: {args.ann_path}")
        print(f"QANN Model: {args.qann_path}")
        
        if BACKWARD_COMPAT:
            print("✓ Backward compatibility verified")
        else:
            print("⚠️  Some backward compatibility issues detected")
    
    # Prepare test data
    if not args.quiet:
        print("\n📊 Preparing CIFAR-10 test data...")
    
    test_set = build_test_dataset()
    test_loader = create_test_dataloader(test_set, args.batch_size)
    
    if not args.quiet:
        print(f"Test set size: {len(test_set)} samples")
    
    # Execute conversion pipeline
    verbose = not args.quiet
    results = run_conversion_pipeline(test_loader, args.ann_path, args.qann_path, verbose)
    
    # Display results
    if results and len(results) > 1:
        display_results(results)
    else:
        print("❌ Pipeline execution failed or incomplete")
        return 1
    
    if not args.quiet:
        print("\n✅ Conversion example completed successfully!")
        print("\nThis example demonstrates the modular ST-BIF framework's")
        print("ability to convert standard ANNs to efficient SNNs while")
        print("maintaining competitive accuracy with improved energy efficiency.")
    
    return 0


if __name__ == "__main__":
    exit(main())