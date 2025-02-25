import torch
import time
from ifneuron_models import OriginalIFNeuron, OptimizedIFNeuron, CudaIFNeuron
from torch.profiler import profile, ProfilerActivity

# +++ 新增可视化需要的import +++
import matplotlib.pyplot as plt

def build_sparse_inputs(batch_size, num_features, timesteps=1, sparsity=0.0, device="cuda"):
    inputs_list = []
    for _ in range(timesteps):
        data = torch.randn(batch_size, num_features, dtype=torch.float32, device=device)
        if sparsity > 0:
            mask = torch.rand_like(data)
            data[mask < sparsity] = 0.0
        inputs_list.append(data)
    return inputs_list

def benchmark_single_step(model, input_tensor, repeat=10):
    model.reset()
    # warm-up
    for _ in range(5):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeat):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / repeat

def benchmark_multi_step(model, inputs_list, repeat=10):
    model.reset()
    # warm-up
    for _ in range(5):
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

def profile_single_step(model, input_tensor, description=""):
    """
    使用 torch.profiler.profile 对 single-step 的 forward 进行分析。
    返回概要表格字符串，并在profile对象中保留详细信息。
    """
    model.reset()
    # warm-up
    for _ in range(5):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    model.reset()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5),
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for _ in range(10):
            _ = model(input_tensor)
            torch.cuda.synchronize()
            prof.step()
            model.reset()
        

    prof_results_table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=10
    )
    result_str = f"\n[Profile Single-Step: {description}]\n" + prof_results_table

    return result_str, prof  # 返回prof对象，可用于可视化

def profile_multi_step(model, inputs_list, description=""):
    """
    使用 torch.profiler.profile 对 multi-step 的 forward 进行分析。
    返回概要表格字符串，并在profile对象中保留详细信息。
    """
    model.reset()
    # warm-up
    for _ in range(5):
        for x_t in inputs_list:
            _ = model(x_t)
    torch.cuda.synchronize()

    model.reset()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5),
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for _ in range(10):
            for x_t in inputs_list:
                _ = model(x_t)
            torch.cuda.synchronize()
            prof.step()
            model.reset()

    prof_results_table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=10
    )
    result_str = f"\n[Profile Multi-Step: {description}]\n" + prof_results_table
    return result_str, prof

def visualize_profile(prof, metric="self_cpu_time_total", top_k=10, output_filename="profile.png", title="Profile Visualization"):
    """
    从给定的profile对象中，提取算子级别信息，做简单的柱状图可视化。
    """
    stats = prof.key_averages()
    # 按照指定 metric 排序，descending
    stats = sorted(stats, key=lambda evt: getattr(evt, metric), reverse=True)

    # 只取前 top_k
    top_stats = stats[:top_k]
    
    # 准备绘图数据
    op_names = [s.key for s in top_stats]
    values = [getattr(s, metric) for s in top_stats]  # 例如 s.self_cpu_time_total

    plt.figure(figsize=(10, 6))
    plt.barh(op_names, values, color="skyblue")
    plt.xlabel(f"{metric} (us)")
    plt.title(title)
    plt.gca().invert_yaxis()  # 让最高的排在上面
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"[Visualization] Saved bar plot to {output_filename}")

def visualize_all_in_one(profiles, top_k=10, output_filename="combined_profile.png"):
    """
    在一张大图里，按顺序把 6 个 Profile 的 CPU耗时 和 GPU耗时 各画一个subplot，共12个子图 (6行2列)。
    其中:
      - 第 i 行 (0<=i<6) 对应第 i 个 profile
      - 左列是 self_cpu_time_total 的柱状图
      - 右列是 self_cuda_time_total 的柱状图
    Args:
        profiles (list[tuple]): [(prof_obj, title_str), ...] 共有6个
        top_k (int): 显示前top_k个算子
        output_filename (str): 输出的单张图文件名
    """
    import math

    assert len(profiles) == 6, "此函数假定恰好有6个profile对象 (3模型x单步/多步)。可按需自行修改。"
    
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(14, 25))
    fig.suptitle("All Profiles (CPU & GPU) in One Figure", fontsize=16)

    # 小函数：在指定的 axes 上画单个柱状图
    def _draw_bar(ax, prof, metric, top_k, subplot_title):
        stats = prof.key_averages()
        stats = sorted(stats, key=lambda e: getattr(e, metric), reverse=True)
        top_stats = stats[:top_k]

        def shorten(name, max_len=30):
            return (name[:max_len] + "...") if len(name) > max_len else name

        op_names = [shorten(s.key, 30) for s in top_stats]
        values = [getattr(s, metric) for s in top_stats]

        ax.barh(op_names, values, color="steelblue")
        ax.set_title(subplot_title, fontsize=10)
        ax.set_xlabel(f"{metric} (us)")
        ax.invert_yaxis()


    # DEBUG: print out atrributes of FunctionEventAvg
    print(torch.__version__)
    print(dir(profiles[0][0].key_averages()[0]))

    for i, (prof_obj, desc_str) in enumerate(profiles):
        # i行，左列 = CPU
        _draw_bar(axes[i, 0], prof_obj, "self_cpu_time_total", top_k, f"[CPU] {desc_str}")
        # i行，右列 = CUDA
        _draw_bar(axes[i, 1], prof_obj, "self_device_time_total", top_k, f"[GPU] {desc_str}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 留出上方标题空间
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"[Visualization] Saved ALL subplots to {output_filename}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_threshold = 0.01
    level = torch.tensor(16)
    sym = False

    batch_size = 1024
    num_features = 1024
    sparsity = 0.1
    timesteps = 16

    # 构建三种模型
    original_model = OriginalIFNeuron(q_threshold, level, sym=sym).to(device)
    optimized_model = OptimizedIFNeuron(q_threshold, level, sym=sym).to(device)
    cuda_model = CudaIFNeuron(q_threshold, level, sym=sym).to(device)

    # =============== Benchmark（原有功能） ===============
    input_tensor = build_sparse_inputs(batch_size, num_features, timesteps=1, sparsity=sparsity, device=device)[0]
    t_orig_1 = benchmark_single_step(original_model, input_tensor, repeat=10)
    t_opt_1 = benchmark_single_step(optimized_model, input_tensor, repeat=10)
    t_cuda_1 = benchmark_single_step(cuda_model, input_tensor, repeat=10)

    inputs_seq = build_sparse_inputs(batch_size, num_features, timesteps=timesteps, sparsity=sparsity, device=device)
    t_orig_seq = benchmark_multi_step(original_model, inputs_seq, repeat=10)
    t_opt_seq = benchmark_multi_step(optimized_model, inputs_seq, repeat=10)
    t_cuda_seq = benchmark_multi_step(cuda_model, inputs_seq, repeat=10)

    print("=== Benchmark Results ===")
    print("[Single-step]")
    print(f"OriginalIFNeuron: {t_orig_1*1e3:.3f} ms")
    print(f"OptimizedIFNeuron: {t_opt_1*1e3:.3f} ms")
    print(f"CudaIFNeuron:     {t_cuda_1*1e3:.3f} ms")

    print("\n[Multi-step]")
    print(f"OriginalIFNeuron (seq): {t_orig_seq*1e3:.3f} ms for {timesteps} steps")
    print(f"OptimizedIFNeuron(seq): {t_opt_seq*1e3:.3f} ms for {timesteps} steps")
    print(f"CudaIFNeuron(seq):     {t_cuda_seq*1e3:.3f} ms for {timesteps} steps")

    # =============== Profile + 文本输出 ===============
    profile_log = []

    # (1) OriginalIF
    single_step_profile_orig_str, single_step_profile_orig_obj = profile_single_step(original_model, input_tensor, "OriginalIFNeuron Single-Step")
    multi_step_profile_orig_str, multi_step_profile_orig_obj = profile_multi_step(original_model, inputs_seq, "OriginalIFNeuron Multi-Step")
    
    

    # (2) OptimizedIF
    single_step_profile_opt_str, single_step_profile_opt_obj = profile_single_step(optimized_model, input_tensor, "OptimizedIFNeuron Single-Step")
    multi_step_profile_opt_str, multi_step_profile_opt_obj = profile_multi_step(optimized_model, inputs_seq, "OptimizedIFNeuron Multi-Step")
    
    

    # (3) CudaIF
    single_step_profile_cuda_str, single_step_profile_cuda_obj = profile_single_step(cuda_model, input_tensor, "CudaIFNeuron Single-Step")
    multi_step_profile_cuda_str, multi_step_profile_cuda_obj = profile_multi_step(cuda_model, inputs_seq, "CudaIFNeuron Multi-Step")
    
    

    profile_log.append(single_step_profile_orig_str)
    profile_log.append(single_step_profile_opt_str)
    profile_log.append(single_step_profile_cuda_str)

    profile_log.append(multi_step_profile_orig_str)
    profile_log.append(multi_step_profile_opt_str)
    profile_log.append(multi_step_profile_cuda_str)

    # 写文本到本地文件
    with open("profile_results.txt", "w", encoding="utf-8") as f:
        for item in profile_log:
            f.write(item)
            f.write("\n\n")
    print("\n[Profiler results have been written to 'profile_results.txt']")

    # =============== 可视化：单独生成小图（原有示例），你可以保留或注释 ===============
    # 例：只演示 OriginalIFNeuron Single-Step
    visualize_profile(single_step_profile_orig_obj,
                      metric="self_cpu_time_total",
                      top_k=10,
                      output_filename="orig_single_step_cpu.png",
                      title="OriginalIF - SingleStep CPU (top10)")

    # =============== 可视化：合并到一张图，6行2列(左CPU+右GPU) ===============
    # 按照你在 txt 里“Single-Step顺序 -> Multi-Step顺序”的排列，依次放置
    # 注：desc_str 仅用于子图标题
    all_profiles = [
        (single_step_profile_orig_obj, "OriginalIFNeuron Single-Step"),
        (single_step_profile_opt_obj,  "OptimizedIFNeuron Single-Step"),
        (single_step_profile_cuda_obj, "CudaIFNeuron Single-Step"),
        (multi_step_profile_orig_obj,  "OriginalIFNeuron Multi-Step"),
        (multi_step_profile_opt_obj,   "OptimizedIFNeuron Multi-Step"),
        (multi_step_profile_cuda_obj,  "CudaIFNeuron Multi-Step"),
    ]

    visualize_all_in_one(
        profiles=all_profiles,
        top_k=10,
        output_filename="all_in_one_profiles.png"
    )
    # 这样就只会生成1张图，共12个小subplot，每一行对应一个Profile：左CPU右GPU

    print("\n[All-in-one visualization saved to 'all_in_one_profiles.png']")
