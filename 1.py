import torch
import time
from ifneuron_models import OriginalIFNeuron, OptimizedIFNeuron, CudaIFNeuron

def build_sparse_inputs(batch_size, num_features, timesteps=1, sparsity=0.0, device="cuda"):
    """
    构造稀疏输入:
    - timesteps: 表示一共构造多少个时刻的输入
    - sparsity: 表示有多少比例的元素为 0 (0.0 表示全密集; 0.9 表示 90% 的元素置0)
    - 返回一个长度为 timesteps 的列表，每个元素是 [batch_size, num_features] 的张量
    """
    inputs_list = []
    for t in range(timesteps):
        data = torch.randn(batch_size, num_features, dtype=torch.float32, device=device)
        if sparsity > 0:
            # 生成一个同形状的掩码，按照 sparsity 的比例置 0
            mask = torch.rand_like(data)
            data[mask < sparsity] = 0.0
        inputs_list.append(data)
    return inputs_list

def compare_outputs(model_a, model_b, input_tensor):
    """
    比较两个模型的单次 forward 输出是否一致。
    返回最大绝对差。
    """
    model_a.reset()
    model_b.reset()

    # 运行
    with torch.no_grad():
        out_a = model_a(input_tensor)
        out_b = model_b(input_tensor)

    max_abs_diff = (out_a - out_b).abs().max().item()
    return max_abs_diff

def compare_sequence(model_a, model_b, inputs_list):
    """
    比较两个模型在多步输入下的输出是否一致。
    输入:
      - inputs_list: [ [batch_size, num_features], [batch_size, num_features], ...]
    输出:
      - 每个 timestep 的输出差异 max_abs_diff 的列表
      - 以及所有时刻最大差异
    """
    model_a.reset()
    model_b.reset()

    diffs = []
    with torch.no_grad():
        for x_t in inputs_list:
            out_a = model_a(x_t)
            out_b = model_b(x_t)
            diff = (out_a - out_b).abs().max().item()
            diffs.append(diff)
    return diffs, max(diffs)

def benchmark_model(model, input_tensor, repeat=1):
    """
    简单的 Benchmark:
      - 重复调用 model.forward(input_tensor) repeat 次
      - 返回平均时长 (秒)
    """
    model.reset()

    # warm-up
    for _ in range(10):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeat):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / repeat

def benchmark_sequence(model, inputs_list, repeat=1):
    """
    对多步输入的 Benchmark：
      - 在 timesteps 步的输入序列上，重复 forward repeat 次
      - 这里的含义是: 每次 forward 都顺序执行 timesteps 步
    """
    model.reset()

    # warm-up
    for _ in range(10):
        for x_t in inputs_list:
            _ = model(x_t)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeat):
        model.reset()
        for x_t in inputs_list:
            _ = model(x_t)
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / repeat

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 神经元参数
    q_threshold = 0.01
    level = torch.tensor(16)  # example
    sym = False

    # 构建模型
    original_model = OriginalIFNeuron(q_threshold, level, sym=sym).to(device)
    optimized_model = OptimizedIFNeuron(q_threshold, level, sym=sym).to(device)
    cuda_model = CudaIFNeuron(q_threshold, level, sym=sym).to(device)

    print("=== Single-step Test ===")
    batch_size = 1024
    num_features = 1024
    sparsity = 0.1
    inputs_single = build_sparse_inputs(batch_size, num_features, timesteps=1, sparsity=sparsity, device=device)
    input_tensor = inputs_single[0]

    # Compare original vs optimized
    max_diff_o_opt = compare_outputs(original_model, optimized_model, input_tensor)
    print(f"Original vs Optimized, single-step max diff = {max_diff_o_opt}")

    # Compare optimized vs cuda
    max_diff_opt_cuda = compare_outputs(optimized_model, cuda_model, input_tensor)
    print(f"Optimized vs CudaIFNeuron, single-step max diff = {max_diff_opt_cuda}")

    # Benchmark
    t_orig = benchmark_model(original_model, input_tensor, repeat=10)
    t_opt = benchmark_model(optimized_model, input_tensor, repeat=10)
    t_cuda = benchmark_model(cuda_model, input_tensor, repeat=10)

    print(f"OriginalIFNeuron: avg forward time = {t_orig*1e3:.3f} ms")
    print(f"OptimizedIFNeuron: avg forward time = {t_opt*1e3:.3f} ms")
    print(f"CudaIFNeuron:     avg forward time = {t_cuda*1e3:.3f} ms")

    print("\n=== Multi-step Test ===")
    timesteps = 16
    inputs_seq = build_sparse_inputs(batch_size, num_features, timesteps=timesteps, sparsity=sparsity, device=device)

    # Compare multiple steps
    diffs_o_opt_list, max_diff_o_opt_seq = compare_sequence(original_model, optimized_model, inputs_seq)
    print(f"[Original vs Optimized] Per-step diffs: {diffs_o_opt_list}, max = {max_diff_o_opt_seq}")

    diffs_opt_cuda_list, max_diff_opt_cuda_seq = compare_sequence(optimized_model, cuda_model, inputs_seq)
    print(f"[Optimized vs Cuda] Per-step diffs: {diffs_opt_cuda_list}, max = {max_diff_opt_cuda_seq}")

    # Benchmark multi-step
    t_orig_seq = benchmark_sequence(original_model, inputs_seq, repeat=10)
    t_opt_seq = benchmark_sequence(optimized_model, inputs_seq, repeat=10)
    t_cuda_seq = benchmark_sequence(cuda_model, inputs_seq, repeat=10)
    print(f"OriginalIFNeuron seq: avg time = {t_orig_seq*1e3:.3f} ms (for {timesteps} steps per forward)")
    print(f"OptimizedIFNeuron seq: avg time = {t_opt_seq*1e3:.3f} ms (for {timesteps} steps per forward)")
    print(f"CudaIFNeuron seq: avg time = {t_cuda_seq*1e3:.3f} ms (for {timesteps} steps per forward)")
