# -*- coding: utf-8 -*-
import os
from pathlib import Path
import importlib.resources as pkg_resources

import torch
from torch.utils.cpp_extension import load as _load_ext


class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    """
    Autograd 封装，外部 API 与原实现保持一致：
    forward(x_seq, v_th, T_max, T_min, prefire) -> (spike_seq[T], v, T_seq[T])
    其中内部仍构造长度 T+1 的缓冲（用于 backward 保存）。
    """
    _built = False
    _ext_mod = None

    @staticmethod
    def _find_cuda_source() -> Path:
        candidates = []
        here = Path(__file__).resolve().parent
        candidates.append(here / "cuda_snn_kernels_new.cu")
        candidates.append(here / "neuron_cupy" / "cuda_snn_kernels_new.cu")

        env_path = os.getenv("CUDA_SNN_KERNELS_PATH")
        if env_path:
            candidates.append(Path(env_path))

        try:
            resource_path = pkg_resources.files("neuron_cupy") / "cuda_snn_kernels_new.cu"
            candidates.append(resource_path)
        except ModuleNotFoundError:
            pass

        for p in candidates:
            try:
                if Path(p).is_file():
                    return Path(p)
            except Exception:
                continue

        tried = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError("未找到 cuda_snn_kernels_new.cu，已尝试：\n  - " + tried)

    @staticmethod
    def _ensure_built():
        if ST_BIFNodeATGF_MS_CUDA._built:
            return
        src = ST_BIFNodeATGF_MS_CUDA._find_cuda_source()

        # 编译并加载，模块初始化时在 C++ 侧通过 TORCH_LIBRARY 完成算子注册
        mod = _load_ext(
            name="snn_cuda_ext",
            sources=[str(src)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ],
            verbose=False,
        )
        ST_BIFNodeATGF_MS_CUDA._ext_mod = mod
        ST_BIFNodeATGF_MS_CUDA._built = True

    @staticmethod
    def forward(ctx,
                x_seq: torch.Tensor,
                v_th: torch.Tensor,
                T_max: torch.Tensor,
                T_min: torch.Tensor,
                prefire: torch.Tensor):

        ST_BIFNodeATGF_MS_CUDA._ensure_built()

        if not x_seq.is_cuda:
            raise RuntimeError("x_seq 必须是 CUDA Tensor")

        # 统一 dtype & device
        dtype = x_seq.dtype
        device = x_seq.device
        v_th = v_th.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        T_max = T_max.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        T_min = T_min.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        prefire = prefire.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        x_seq = x_seq.contiguous()

        # 调用自定义 CUDA op：返回 (spike_all[T+1], v, T_all[T+1], H_all[T+1])
        spike_all, v_out, T_all, H_all = torch.ops.snn.st_bif_forward(
            x_seq, v_th, T_max, T_min, prefire
        )

        # 保存用于反传；对外仍返回切片后的 (T, ...)
        ctx.save_for_backward(spike_all, T_all, H_all, v_th, T_max, T_min)
        return spike_all[1:], v_out, T_all[1:]

    @staticmethod
    def backward(ctx,
                 grad_spike_seq: torch.Tensor,
                 grad_v: torch.Tensor,
                 grad_T_seq: torch.Tensor):
        spike_all, T_all, H_all, v_th, T_max, T_min = ctx.saved_tensors

        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v = grad_v.contiguous()
        grad_T_seq = grad_T_seq.contiguous()

        # 传入保存的 (T+1) 缓冲，CUDA op 内部完成时间反向扫描
        grad_x = torch.ops.snn.st_bif_backward(
            grad_spike_seq, grad_v, grad_T_seq,
            spike_all, T_all, H_all,
            v_th, T_max, T_min
        )
        return grad_x, None, None, None, None


# 便捷函数：与原先的 apply 行为一致
def st_bifnode_atgf_ms(x_seq, v_th, T_max, T_min, prefire):
    return ST_BIFNodeATGF_MS_CUDA.apply(x_seq, v_th, T_max, T_min, prefire)


if __name__ == "__main__":
    x_seq = torch.randn(10, 10, 10).to("cuda")
    v_th = torch.randn(1).to("cuda")
    T_max = torch.randn(1).to("cuda")
    T_min = torch.randn(1).to("cuda")
    prefire = torch.randn(1).to("cuda") 
    spike_seq, v_out, T_seq = st_bifnode_atgf_ms(x_seq, v_th, T_max, T_min, prefire)
    print(spike_seq, v_out, T_seq)