#!/usr/bin/env python3
# cuda_kernel_profiler_v3.2.py  ──────────────────────────────────────────
#
# 比较 ST-BIF CUDA kernels（orig vs. new）在多精度 / timestep / feature
# 上的 forward & backward 性能；输出 JSON 与 bar-plot PNG。

from __future__ import annotations
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

# ─── Project path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # 调整到你的 repo 层级
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

# ─── 内核可用性检测 ───────────────────────────────────────────────────────
try:
    from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA as OriginalKernel
    original_available = True
except Exception as e:
    print(f"\033[33m[Warn] 原始内核不可用: {e}\033[0m")
    original_available = False

try:
    from neuron_cupy.cuda_operator_new import ST_BIFNodeATGF_MS_CUDA as NewKernel
    new_available = True
except Exception as e:
    print(f"\033[33m[Warn] 新内核不可用: {e}\033[0m")
    new_available = False

from snn.neurons.st_bif_neurons import ST_BIFNeuron_MS


# ════════════════════════════════════════════════════════════════════════
class CUDAKernelProfiler:
    _ATOL = {"fp64": 1e-7, "fp32": 1e-5, "fp16": 1e-3}
    _RTOL = {"fp64": 1e-5, "fp32": 1e-4, "fp16": 1e-2}
    _DTYPE = {"fp64": torch.float64, "fp32": torch.float32, "fp16": torch.float16}

    def __init__(self, bs: int, runs: int, feats: List[int], ts_list: List[int],
                 precisions: List[str], include_backward: bool, out_dir: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA 设备")

        self.bs, self.runs = bs, runs
        self.feats, self.ts_list = feats, ts_list
        self.precisions = precisions
        self.do_backward = include_backward
        self.dev = "cuda"
        self.out_dir = out_dir

        self._data_cache: Dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self.records: List[Dict] = []

        self.kernels = []
        if original_available:
            self.kernels.append("orig")
        if new_available:
            self.kernels.append("new")
        if not self.kernels:
            raise RuntimeError("找不到可用内核")

    # ─── 输入缓存 ───────────────────────────────────────────────────────
    def _get_data(self, F: int, dtype: torch.dtype, max_T: int):
        key = (F, dtype)
        if key not in self._data_cache:
            torch.manual_seed(42)
            self._data_cache[key] = torch.randn(self.bs * max_T, F,
                                                device=self.dev, dtype=dtype)
        return self._data_cache[key]

    # ─── 构造神经元 ────────────────────────────────────────────────────
    def _build_neuron(self, T: int, dtype: torch.dtype, use_new: bool):
        n = ST_BIFNeuron_MS(torch.tensor(1.0, dtype=dtype, device=self.dev),
                            level=8, sym=True, first_neuron=True).to(dtype)
        n.T = T

        if use_new:
            import types

            def fwd(self, x):
                N = x.size(0)
                spk, _, _ = NewKernel.apply(
                    x.view(T, N // T, -1).flatten(2),
                    self.q_threshold, self.pos_max, self.neg_min, self.prefire
                )
                return spk.view_as(x) * self.q_threshold

            n.forward = types.MethodType(fwd, n)

        return n.eval().to(self.dev)

    # ─── 等效性检查 ────────────────────────────────────────────────────
    def _equiv(self, T: int, F: int, prec: str):
        if not (original_available and new_available):
            return

        dtype = self._DTYPE[prec]
        atol, rtol = self._ATOL[prec], self._RTOL[prec]

        x = self._get_data(F, dtype, T)[:self.bs * T].clone()

        a = self._build_neuron(T, dtype, use_new=False)
        b = self._build_neuron(T, dtype, use_new=True)

        with torch.inference_mode():
            a.reset(); oa = a(x)
            b.reset(); ob = b(x)

        diff = (oa - ob).abs()
        md, mean = diff.max().item(), diff.mean().item()
        ok = md <= atol or md <= rtol * oa.abs().max().item()
        tag = "\033[32m✓" if ok else "\033[31m✗"
        print(f"      等效性 {tag} (max={md:.2e}, mean={mean:.2e})")

    # ─── Profile 单 kernel / direction ────────────────────────────────
    def _profile(self, T: int, F: int, prec: str, kernel: str, direction: str):
        dtype = self._DTYPE[prec]
        use_new = (kernel == "new")
        x_full = self._get_data(F, dtype, T)
        x = x_full[:self.bs * T].clone().requires_grad_(direction == "backward")

        n = self._build_neuron(T, dtype, use_new)

        # 预热
        with torch.inference_mode():
            for _ in range(4):
                n.reset(); _ = n(x)
        if direction == "backward":
            n.reset(); (n(x)).sum().backward(); torch.cuda.synchronize()

        torch.cuda.synchronize()
        times = []
        memory_peaks = []
        
        for i in range(self.runs):
            nvtx.range_push(f"{kernel}_{direction}_{i}")
            n.reset()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 记录执行前内存
            mem_before = torch.cuda.memory_allocated()
            t0 = time.perf_counter()

            if direction == "forward":
                with torch.inference_mode():
                    _ = n(x)
            else:
                (n(x)).sum().backward()

            torch.cuda.synchronize()
            # 记录执行后内存峰值
            mem_peak = torch.cuda.max_memory_allocated()
            memory_peaks.append(mem_peak - mem_before)
            torch.cuda.reset_peak_memory_stats()
            
            times.append((time.perf_counter() - t0) * 1e3)
            nvtx.range_pop()

        arr = np.asarray(times)
        mem_arr = np.asarray(memory_peaks)
        mean_ms = arr.mean().item()
        mean_mem_mb = mem_arr.mean().item() / (1024**2)  # 转换为MB

        print(f"      {kernel:<4s}-{direction[0]} : {mean_ms:7.2f} ms "
              f"({mean_mem_mb:6.1f} MB)")

        self.records.append(dict(
            precision=prec, kernel=kernel, direction=direction,
            T=T, F=F, metric="mean_ms", value=mean_ms
        ))
        self.records.append(dict(
            precision=prec, kernel=kernel, direction=direction,
            T=T, F=F, metric="memory_mb", value=mean_mem_mb
        ))

    # ─── 主循环 ────────────────────────────────────────────────────────
    def run(self):
        print(f"\nCUDA Kernel Profiler │ device: {torch.cuda.get_device_name(0)}")
        print(f"bs={self.bs}, runs={self.runs}, precisions={self.precisions}, "
              f"Ts={self.ts_list}, Fs={self.feats}, backward={self.do_backward}\n")

        for prec in self.precisions:
            print(f"── 精度 {prec} ───────────────────────────────────")
            for T in self.ts_list:
                for F in self.feats:
                    print(f"    T={T:2d}, F={F:4d}")
                    self._equiv(T, F, prec)

                    for kernel in self.kernels:
                        self._profile(T, F, prec, kernel, "forward")
                        if self.do_backward:
                            self._profile(T, F, prec, kernel, "backward")

    # ─── 保存 JSON + barplot PNG ──────────────────────────────────────
        # ─── 保存 JSON + barplot PNG ──────────────────────────────────────
    def save(self):
        import json, pandas as pd, matplotlib.pyplot as plt

        df = pd.DataFrame(self.records)
        js = self.out_dir / "kernel_prof_full.json"
        js.write_text(json.dumps(df.to_dict("records"), indent=2))
        print(f"\n完整结果已保存: {js}")

        if len(self.kernels) < 2:
            print("仅检测到一个内核，跳过 barplot 对比")
            return

        # -- 绘制1-2张综合对比图 ----------------------------
        # 图1: Latency 对比 (所有精度和方向)
        latency_data = df[df["metric"] == "mean_ms"]
        if not latency_data.empty:
            fig, axes = plt.subplots(1, len(self.precisions), figsize=(6*len(self.precisions), 5))
            if len(self.precisions) == 1:
                axes = [axes]
            
            for i, prec in enumerate(self.precisions):
                sub_prec = latency_data[latency_data["precision"] == prec]
                if sub_prec.empty:
                    continue
                    
                # 组合标签：T?-F?-方向
                sub_prec = sub_prec.copy()
                sub_prec["cfg"] = ("T" + sub_prec["T"].astype(str) + 
                                  "-F" + sub_prec["F"].astype(str) + 
                                  "-" + sub_prec["direction"].str[:3])
                
                # 透视表
                pivot = (sub_prec.pivot_table(index="cfg", columns="kernel", values="value")
                        .reindex(sorted(sub_prec["cfg"].unique())))
                
                pivot.plot(kind="bar", ax=axes[i])
                axes[i].set_title(f"{prec.upper()} Latency (ms)")
                axes[i].set_xlabel("T-F-方向")
                axes[i].set_ylabel("Latency (ms)")
                axes[i].grid(True, linestyle="--", alpha=.4)
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.out_dir / "latency_comparison.png", dpi=120)
            plt.close()

        # 图2: GPU Memory 对比 (所有精度和方向)
        memory_data = df[df["metric"] == "memory_mb"]
        if not memory_data.empty:
            fig, axes = plt.subplots(1, len(self.precisions), figsize=(6*len(self.precisions), 5))
            if len(self.precisions) == 1:
                axes = [axes]
            
            for i, prec in enumerate(self.precisions):
                sub_prec = memory_data[memory_data["precision"] == prec]
                if sub_prec.empty:
                    continue
                    
                # 组合标签：T?-F?-方向
                sub_prec = sub_prec.copy()
                sub_prec["cfg"] = ("T" + sub_prec["T"].astype(str) + 
                                  "-F" + sub_prec["F"].astype(str) + 
                                  "-" + sub_prec["direction"].str[:3])
                
                # 透视表
                pivot = (sub_prec.pivot_table(index="cfg", columns="kernel", values="value")
                        .reindex(sorted(sub_prec["cfg"].unique())))
                
                pivot.plot(kind="bar", ax=axes[i])
                axes[i].set_title(f"{prec.upper()} GPU Memory (MB)")
                axes[i].set_xlabel("T-F-方向")
                axes[i].set_ylabel("Memory Usage (MB)")
                axes[i].grid(True, linestyle="--", alpha=.4)
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.out_dir / "memory_comparison.png", dpi=120)
            plt.close()

        print(f"对比图表已保存: latency_comparison.png, memory_comparison.png")

# ─── CLI ────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser(
        description="Profile orig/new ST-BIF CUDA kernels with multi-precision."
    )
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--features", nargs="+", type=int, default=[256, 512])
    p.add_argument("--ts", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--precisions", nargs="+", choices=["fp64", "fp32", "fp16"],
                   default=["fp64", "fp32", "fp16"])
    p.add_argument("--no_backward", action="store_true",
                   help="Skip backward profiling")
    return p.parse_args()


def main():
    args = parse()
    ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "nsys_results" / ts_name
    out_dir.mkdir(parents=True, exist_ok=True)

    profiler = CUDAKernelProfiler(
        bs=args.bs, runs=args.runs,
        feats=args.features, ts_list=args.ts,
        precisions=args.precisions,
        include_backward=not args.no_backward,
        out_dir=out_dir
    )
    profiler.run()
    profiler.save()


if __name__ == "__main__":
    main()
