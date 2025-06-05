#!/usr/bin/env python3
# cuda_kernel_profiler.py  ────────────────────────────────────────────────
# Compare ST-BIF CUDA kernels (original vs. new)

from __future__ import annotations
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

# ─── Project path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="cupy")

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
    def __init__(self, bs: int, T: int, runs: int):
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA 设备")

        self.bs, self.T, self.runs = bs, T, runs
        self.dev = "cuda"
        self._data_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        self.results: Dict = {
            "config": {
                "batch_size": bs, "time_steps": T, "num_runs": runs,
                "device": torch.cuda.get_device_name(0),
                "timestamp": datetime.now().isoformat()
            },
            "kernels": {}, "comparison": {}
        }

    # ── 数据 ────────────────────────────────────────────────────────────
    def _get_data(self, F: int):
        if F not in self._data_cache:
            torch.manual_seed(42)
            self._data_cache[F] = {
                "x_seq": torch.randn(self.T * self.bs, F, device=self.dev)
            }
        return self._data_cache[F]

    # ── 构造神经元 ───────────────────────────────────────────────────────
    def _build_neuron(self, use_new: bool):
        n = ST_BIFNeuron_MS(torch.tensor(1.0), level=8,
                            sym=True, first_neuron=True).to(self.dev)
        n.T = self.T
        if use_new:
            import types
            def fwd(self, x):
                N = x.size(0)
                spk, _, _ = NewKernel.apply(
                    x.view(self.T, N//self.T, -1).flatten(2),
                    self.q_threshold, self.pos_max, self.neg_min, self.prefire
                )
                return spk.view_as(x) * self.q_threshold
            n.forward = types.MethodType(fwd, n)
        return n.eval()

    # ── 等效性 ───────────────────────────────────────────────────────────
    def _equiv(self, F: int, atol=1e-5, rtol=1e-4, show=8):
        if not (original_available and new_available):
            return
        data = self._get_data(F)
        a, b = self._build_neuron(False), self._build_neuron(True)
        with torch.inference_mode():
            a.reset(); oa = a(data["x_seq"])
            b.reset(); ob = b(data["x_seq"])
        diff = (oa - ob).abs()
        md, mean = diff.max().item(), diff.mean().item()
        rel = mean / (oa.abs().mean().item() + 1e-8)
        ok = md <= atol or md <= rtol * oa.abs().max().item()
        tag = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"  等效性 {tag}  (max={md:.2e}, mean={mean:.2e}, rel={rel:.2e})")

        self.results["comparison"][f"F{F}"] = {
            "equivalent": ok, "max_diff": md,
            "mean_diff": mean, "rel_err": rel
        }

        if ok:
            return
        idx = torch.nonzero(diff > atol, as_tuple=False)
        print(f"    ➜ 首 {min(show, idx.size(0))} 处不一致 (t,b,f,val_o,val_n):")
        for j in range(min(show, idx.size(0))):
            if diff.dim() == 3:               # (T,B,F)
                t, b, f = idx[j].tolist()
            else:                             # (T*B,F) → 映射回 (t,b)
                tb, f  = idx[j].tolist()
                t, b   = divmod(tb, self.bs)  # ← 修正
            val_o = oa.view(-1, oa.size(-1))[idx[j, 0], f].item()
            val_n = ob.view(-1, ob.size(-1))[idx[j, 0], f].item()
            print(f"      ({t},{b},{f}) {val_o:+.3e} vs {val_n:+.3e}")

    # ── 性能 ────────────────────────────────────────────────────────────
    def _profile(self, name: str, use_new: bool, F: int):
        data  = self._get_data(F)
        n     = self._build_neuron(use_new)
        with torch.inference_mode():
            for _ in range(5):
                n.reset(); _ = n(data["x_seq"])
        torch.cuda.synchronize()
        times = []
        for i in range(self.runs):
            n.reset(); nvtx.range_push(f"{name}_{i}")
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = n(data["x_seq"])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            nvtx.range_pop()
        arr = np.asarray(times)
        thr = self.bs * self.T / (arr.mean() / 1000)
        print(f"  {name}: {arr.mean():.3f} ms  p90 {np.percentile(arr,90):.3f} ms  "
              f"{thr:,.0f} samp/s")
        res = {"timing": {"mean_ms": arr.mean().item(),
                          "p90_ms":  np.percentile(arr, 90).item(),
                          "throughput_sps": thr}}
        return res

    # ── 运行 ─────────────────────────────────────────────────────────────
    def run(self, feats: List[int]):
        print(f"\nCUDA Kernel Profiler  |  device: {self.results['config']['device']}")
        print(f"bs={self.bs},  T={self.T},  runs={self.runs}\n")
        for F in feats:
            print(f"── Feature {F} ─────────────────────────────────────────")
            if original_available and new_available:
                self._equiv(F)

            if original_available:
                self.results["kernels"][f"orig_F{F}"] = \
                    self._profile("Original", False, F)
            if new_available:
                self.results["kernels"][f"new_F{F}"] = \
                    self._profile("New", True,  F)
        out_dir = ROOT / "outputs" / "nsys_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"kernel_prof_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(self.results, indent=2))
        print(f"\n结果已保存: {path}")


# ─── CLI ────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--ts", type=int, default=8)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--features", type=int, nargs="+", default=[256, 512])
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    if not original_available:
        print("原始内核不可用，退出"); sys.exit(1)
    profiler = CUDAKernelProfiler(args.bs, args.ts, args.runs)
    profiler.run(args.features)
