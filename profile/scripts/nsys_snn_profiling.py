#!/usr/bin/env python3
"""
ST-BIF SNN – Nsight Systems NVTX profiling script
-------------------------------------------------
• 只在关键层打 NVTX：SNN_Inference_Session / SNN_Inference_Run_X / Model_Forward_Pass  
• 使用标准PyTorch推理，无CUDA Graph优化，确保profiling准确性
• 不再用 torch.cuda.Event 同步分段；精确时间让 Nsight 统计，脚本仅记录总用时  
• 输入拷贝采用 pinned-memory + non_blocking=True，减少 H→D 等待  
"""

import os, sys, time, json, warnings
from datetime import datetime

import torch
import torch.cuda.nvtx as nvtx
import numpy as np

# ----------------------------------------------------------------------
# 工程内模块
sys.path.append('../..')
from models.resnet import resnet18
from snn.conversion.quantization import myquan_replace_resnet
from wrapper.snn_wrapper import SNNWrapper_MS
from wrapper.encoding import get_subtensors
# ----------------------------------------------------------------------


class NVTXSNNProfiler:
    # ------------------------------ 初始化 ------------------------------
    def __init__(self, batch_size: int = 32, num_runs: int = 5):
        self.batch_size = batch_size
        self.num_runs  = num_runs
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 运行统计
        self.results = {
            "run_times_ms": [],
            "metadata": {
                "batch_size": batch_size,
                "num_runs":   num_runs,
                "device":     torch.cuda.get_device_name(0) if self.device == 'cuda' else "cpu",
                "timestamp":  datetime.now().isoformat()
            }
        }

        # 预分配缓冲区（避免动态分配）
        self._pinned_cpu  = None      # 用于 H→D 异步 copy
        self._T           = None
        self._B           = None

    # ------------------------------ 模型加载 -----------------------------
    def load_snn_model(self):
        print("Loading ANN→SNN ...")
        qann = resnet18(num_classes=10).to(self.device)
        myquan_replace_resnet(qann, level=8, weight_bit=32)
        qann_ckpt = torch.load(
            "/home/zilingwei/Projects/ST-BIF_Profiling/checkpoints/resnet/best_QANN.pth",
            map_location=self.device
        )
        qann.load_state_dict(qann_ckpt)
        qann.eval()

        self.snn_model = SNNWrapper_MS(
            ann_model   = qann,
            time_step   = 8,
            level       = 8,
            neuron_type = "ST-BIF"
        ).to(self.device).eval()

        self._T = self.snn_model.T
        print("✓ SNN ready")

    # ------------------------------ 输入生成 -----------------------------
    def create_input(self):
        x = torch.randn(self.batch_size, 3, 32, 32, device="cpu")  # 先放 CPU，方便演示 pinned copy
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        x = (x * std) + mean
        return x

    # -------------------------- 预热和缓冲区准备 -------------------------
    def _prepare_buffers(self, example_batch: torch.Tensor):
        torch.backends.cudnn.benchmark = True

        # ---------- 首次编码 ----------
        enc_gpu = get_subtensors(
            example_batch.cuda(non_blocking=True), 0.0, 0.0,
            sample_grain=self.snn_model.step,
            time_step   = self._T
        ).contiguous()
        if enc_gpu.dim() == 5:
            self._B = enc_gpu.shape[1]
            enc_gpu = enc_gpu.view(self._T * self._B, *enc_gpu.shape[2:])
        else:
            self._B = self.batch_size

        # ---------- 预热推理 ----------
        print("⇢ Warming up model ...")
        with torch.inference_mode():
            for _ in range(3):  # 多次预热确保稳定
                self.snn_model._reset_all_states()
                _ = self.snn_model.model(enc_gpu)
        torch.cuda.synchronize()

        # ⭐ pinned buffer 形状与 encoded 一致
        self._pinned_cpu = torch.empty_like(enc_gpu, device='cpu', pin_memory=True)
        
        print("✓ Model warmed up\n")



    # ----------------------- 单次推理（含 NVTX） -------------------------
    @torch.inference_mode()
    def snn_forward_nvtx(self, cpu_batch: torch.Tensor, run_idx: int):
        nvtx.range_push(f"SNN_Inference_Run_{run_idx}")

        # -------- 1) SNN 状态重置 ----------
        nvtx.range_push("SNN_State_Reset")
        self.snn_model._reset_all_states()
        nvtx.range_pop()

        # -------- 2) 输入处理 ----------
        nvtx.range_push("Input_Processing")
        
        nvtx.range_push("CPU_to_GPU_Transfer")
        gpu_batch = cpu_batch.cuda(non_blocking=True)
        nvtx.range_pop()
        
        nvtx.range_push("Temporal_Encoding")
        enc_gpu = get_subtensors(
            gpu_batch, 0.0, 0.0,
            sample_grain=self.snn_model.step,
            time_step   = self._T
        ).contiguous()
        nvtx.range_pop()

        if enc_gpu.dim() == 5:                       # [T, B, C, H, W] → [T*B, C, H, W]
            enc_gpu = enc_gpu.view(self._T * self._B, *enc_gpu.shape[2:])
        nvtx.range_pop()  # /Input_Processing

        # -------- 3) 模型前向传播 ----------
        nvtx.range_push("Model_Forward_Pass")
        snn_output = self.snn_model.model(enc_gpu)
        nvtx.range_pop()                             # /Model_Forward_Pass

        # -------- 4) 输出处理 ----------
        nvtx.range_push("Output_Processing")
        out = snn_output.view(self._T, self._B, *snn_output.shape[1:]).sum(dim=0)
        nvtx.range_pop()  # /Output_Processing

        nvtx.range_pop()                             # /SNN_Inference_Run_X
        return out



    # -------------------- Profiling 会话 -------------------------------
    def run(self):
        print("\n==========  Profiling session  ==========")

        # 1. 载模型 & warm-up
        self.load_snn_model()
        warm_batch = self.create_input()

        # warm-up
        if self.device == 'cuda':
            self._prepare_buffers(warm_batch)
        else:
            raise RuntimeError("CUDA required for this profiling script")

        # 2. 正式采样
        nvtx.range_push("SNN_Inference_Session")

        for i in range(self.num_runs):
            cpu_batch = self.create_input()

            start = time.perf_counter()
            _ = self.snn_forward_nvtx(cpu_batch, i)
            torch.cuda.synchronize()          # 等待 GPU 完成
            dt = (time.perf_counter() - start) * 1e3
            self.results["run_times_ms"].append(dt)

            print(f"Run {i+1}/{self.num_runs}  –  {dt:6.2f} ms")

        nvtx.range_pop()  # /SNN_Inference_Session

        # 3. 保存统计
        self._save_results()
        print("\n✓ Done.  Results written to outputs/nsys_results/*")

    # -------------------------- 结果保存 -------------------------------
    def _save_results(self):
        os.makedirs("../outputs/nsys_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        mean_t = np.mean(self.results["run_times_ms"])
        std_t  = np.std(self.results["run_times_ms"])
        thr    = self.batch_size / (mean_t / 1e3)

        summary = {
            "mean_ms":     float(mean_t),
            "std_ms":      float(std_t),
            "min_ms":      float(np.min(self.results["run_times_ms"])),
            "max_ms":      float(np.max(self.results["run_times_ms"])),
            "throughput":  float(thr)
        }
        self.results["summary"] = summary

        json_path = f"../outputs/nsys_results/nsys_profiling_results_{timestamp}.json"
        txt_path  = f"../outputs/nsys_results/nsys_profiling_summary_{timestamp}.txt"

        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        with open(txt_path, "w") as f:
            f.write("ST-BIF SNN - Nsight Systems Profiling\n")
            f.write("=" * 52 + "\n")
            f.write(json.dumps(summary, indent=2))

        print(f"  • JSON   {json_path}")
        print(f"  • TXT    {txt_path}")


# ======================================================================
def main():
    print("SNN NVTX profiler - generates nsys-friendly traces")
    profiler = NVTXSNNProfiler(batch_size=32, num_runs=5)
    profiler.run()

    s = profiler.results["summary"]
    print(f"\nFinal avg: {s['mean_ms']:.2f} ms  |  {s['throughput']:.1f} samples/s")


if __name__ == "__main__":
    main()
