#!/usr/bin/env python3
# 文件名: profile_stbif_forward.py

import torch
from torch.cuda import nvtx
import time

# 假设你已经在同目录下有 ifneuron_models.py
# 并且其中定义了 CudaIFNeuron 类和 stbif_cuda 内核
from ifneuron_models import CudaIFNeuron

def build_random_input(batch_size, num_features, device="cuda"):
    """
    构造一些简单的随机张量作为输入。
    你也可以改成更复杂的输入生成过程
    """
    x = torch.randn(batch_size, num_features, device=device, dtype=torch.float32)
    return x

def main():
    device = "cuda"
    print(f"[INFO] Using device = {device}")

    batch_size = 256
    num_features = 256
    warmup_steps = 5
    profile_steps = 10

    q_threshold = 0.01
    level = torch.tensor(16).to(device) 
    sym = False

    model = CudaIFNeuron(q_threshold, level, sym=sym).to(device)
    model.eval()
    input_tensor = build_random_input(batch_size, num_features, device=device)

    # --------------- Warmup ---------------
    print(f"[INFO] Warmup for {warmup_steps} steps ...")
    for i in range(warmup_steps):
        _ = model(input_tensor)
        torch.cuda.synchronize()
    print("[INFO] Warmup done.\n")

    # --------------- Profile steps ---------------
    print(f"[INFO] Now running {profile_steps} steps for NCU profiling ...")

    with nvtx.range("TotalProfileRegion"):
        for step_idx in range(profile_steps):
            with nvtx.range(f"Inference-Step-{step_idx}"):
                _ = model(input_tensor)
                model.reset()
                torch.cuda.synchronize()

    print("[INFO] Finished profile steps.\n")


    print("[INFO] Script completed.")

if __name__ == "__main__":
    main()
