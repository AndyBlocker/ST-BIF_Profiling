# st_bif_cuda.py
# ================================================================
# 优化版 ST_BIFNodeATGF_MS_CUDA —— 接口完全兼容旧实现
# ================================================================
import os
from pathlib import Path
import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import importlib.resources as pkg_resources

__all__ = ["ST_BIFNodeATGF_MS_CUDA"]


class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    """
    高性能 SNN BIF 节点（多阈值 & Prefire）CUDA 实现

    🔄 接口与旧版完全保持兼容；可直接 `git mv` / 覆盖替换。
    """

    # ------------------------------------------------------------------
    # ① 类级缓存：RawModule & 已绑定内核
    # ------------------------------------------------------------------
    _cuda_source: str | None = None
    _module: cp.RawModule | None = None
    _kernels: dict[str, cp.RawKernel] = {}

    # ------------------------------------------------------------------
    # ② CUDA 源碼加载 / 编译
    # ------------------------------------------------------------------
    @classmethod
    def _load_cuda_kernels(cls) -> None:
        if cls._module is not None:
            return  # 已经编译好

        # -------- 1. 按优先级搜索 kernel 文件 --------------------------
        candidates: list[Path] = []

        # 与当前 .py 同级/子目录
        here = Path(__file__).resolve().parent
        candidates.append(here / "neuron_cupy" / "cuda_snn_kernels_new.cu")

        # 环境变量指向
        env_path = os.getenv("CUDA_SNN_KERNELS_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        # pip install 场景下的包内资源
        try:
            resource_path = pkg_resources.files("neuron_cupy") / "cuda_snn_kernels_new.cu"
            candidates.append(resource_path)
        except ModuleNotFoundError:
            pass

        cuda_code = None
        for p in candidates:
            try:
                if p.is_file():
                    cuda_code = p.read_text(encoding="utf-8")
                    break
            except Exception:
                continue

        if cuda_code is None:
            tried = "\n  - ".join(str(p) for p in candidates)
            raise FileNotFoundError("cuda_snn_kernels_new.cu not found; searched:\n  - " + tried)

        cls._cuda_source = cuda_code

        compute_capability = torch.cuda.get_device_capability()
        arch_flag = f"-arch=sm_{compute_capability[0]}{compute_capability[1]}"

        cls._module = cp.RawModule(
            code=cuda_code,
            name_expressions=[
                "forward_kernel_fp16", "backward_kernel_fp16",
                "forward_kernel_fp32", "backward_kernel_fp32",
                "forward_kernel_fp64", "backward_kernel_fp64",
            ],
            options=(
                "--std=c++17",
                "--use_fast_math",
            ),
        )

        for name in cls._module.name_expressions:
            cls._kernels[name] = cls._module.get_function(name)


    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_seq: torch.Tensor,
        v_th: torch.Tensor,
        T_max: torch.Tensor,
        T_min: torch.Tensor,
        prefire: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x_seq   : (T, N, ...) 输入电流
        v_th    : 标量阈值
        T_max   : 不可再发正脉冲的上限窗口
        T_min   : 不可再发负脉冲的下限窗口
        prefire : 首帧预充电系数
        """
        ST_BIFNodeATGF_MS_CUDA._load_cuda_kernels()

        # ---------- shape & dtype ----------
        T, N, *feat_shape = x_seq.shape
        feature_flat = int(torch.prod(torch.tensor(feat_shape)).item())
        total_neuron = N * feature_flat
        dtype = x_seq.dtype
        device = x_seq.device

        # ---------- 选择内核 ----------
        if dtype == torch.float16:
            k_fwd = ST_BIFNodeATGF_MS_CUDA._kernels["forward_kernel_fp16"]
        elif dtype == torch.float64:
            k_fwd = ST_BIFNodeATGF_MS_CUDA._kernels["forward_kernel_fp64"]
        else:
            k_fwd = ST_BIFNodeATGF_MS_CUDA._kernels["forward_kernel_fp32"]

        # ---------- 输出张量 ----------
        spike_seq = torch.zeros((T + 1, N, *feat_shape), device=device, dtype=dtype)
        T_seq = torch.zeros_like(spike_seq)
        H_seq = torch.zeros_like(spike_seq)
        v_out = torch.empty((N, *feat_shape), device=device, dtype=dtype)

        # ---------- DLPack 转 CuPy ----------
        def t2c(t: torch.Tensor):
            return cp.from_dlpack(to_dlpack(t))

        x_seq_c = t2c(x_seq.contiguous())
        spike_c = t2c(spike_seq)
        v_out_c = t2c(v_out)
        T_seq_c = t2c(T_seq)
        H_seq_c = t2c(H_seq)

        # 标量参数按 dtype 转换后放到 0 维张量再传
        def scalar_as_array(x: torch.Tensor, target_dtype):
            out = x.to(device=device, dtype=target_dtype)
            return t2c(out)

        v_th_c = scalar_as_array(v_th, dtype)
        Tmax_c = scalar_as_array(T_max, dtype)
        Tmin_c = scalar_as_array(T_min, dtype)
        pre_c = scalar_as_array(prefire, dtype)

        # ---------- grid / block ----------
        threads = 128
        # grid-stride 循环，gridDim.x 不必超 65535
        blocks = min((total_neuron + threads - 1) // threads, 65535)

        # ---------- 调 kernel ----------
        #   向 RawKernel 传 Python int 会被自动 cast 为 32‑bit
        k_fwd(
            (blocks,), (threads,),
            (
                x_seq_c,
                v_th_c,
                Tmax_c,
                Tmin_c,
                pre_c,
                spike_c,
                v_out_c,
                T_seq_c,
                H_seq_c,
                N, T, feature_flat, total_neuron
            ),
        )

        # ---------- 存上下文 ----------
        ctx.save_for_backward(spike_seq, T_seq, H_seq, v_th, T_max, T_min)
        ctx.dtype = dtype
        ctx.feature_flat = feature_flat
        ctx.total_neuron = total_neuron

        # 返回 T 个时刻（去掉 t=0）与末状态
        return spike_seq[1:], v_out, T_seq[1:]

    # ------------------------------------------------------------------
    # ④ 反向传播
    # ------------------------------------------------------------------
    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_spike_seq: torch.Tensor,
        grad_v: torch.Tensor,
        grad_T_seq: torch.Tensor,
    ):
        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        dtype = ctx.dtype
        feature_flat = ctx.feature_flat
        total_neuron = ctx.total_neuron

        if dtype == torch.float16:
            k_bwd = ST_BIFNodeATGF_MS_CUDA._kernels["backward_kernel_fp16"]
        elif dtype == torch.float64:
            k_bwd = ST_BIFNodeATGF_MS_CUDA._kernels["backward_kernel_fp64"]
        else:
            k_bwd = ST_BIFNodeATGF_MS_CUDA._kernels["backward_kernel_fp32"]

        # --------- allocate grad_x ---------
        grad_x = torch.zeros_like(grad_spike_seq)

        # --------- DLPack ---------
        def t2c(t): return cp.from_dlpack(to_dlpack(t.contiguous()))

        args = (
            t2c(grad_spike_seq), t2c(grad_v), t2c(grad_T_seq),
            t2c(spike_seq), t2c(T_seq), t2c(H_seq),
            t2c(v_th.to(dtype)), t2c(T_max.to(dtype)), t2c(T_min.to(dtype)),
            t2c(grad_x),
            grad_spike_seq.shape[1],  # N
            grad_spike_seq.shape[0],  # T
            feature_flat, total_neuron
        )

        threads = 256
        blocks = min((total_neuron + threads - 1) // threads, 65535)
        k_bwd((blocks,), (threads,), args)

        # 无其它叶子需要梯度
        return grad_x, None, None, None, None
