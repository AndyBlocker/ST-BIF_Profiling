import cupy as cp
import torch
import os
from pathlib import Path
import importlib.resources as pkg_resources
from torch.utils.dlpack import to_dlpack, from_dlpack


first = True
class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    cuda_source = None  # Will store the CUDA source code
    count = 0
    v_thr1 = 0
    
    @staticmethod
    def _load_cuda_kernels():
        """
        1. 仅在 cuda_source 为空时真正加载/编译，防止重复工作  
        2. 依次尝试以下几种查找策略，直到找到 cuda_snn_kernels.cu：
        • 与当前 .py 文件同级或子目录 ./neuron_cupy/  
        • 环境变量 CUDA_SNN_KERNELS_PATH 指定的绝对 / 相对路径  
        • 安装包资源 neuron_cupy/cuda_snn_kernels.cu（支持 pip install 后的情况）
        3. 找不到时明确抛出 FileNotFoundError，并列出所有尝试过的路径，方便调试
        """
        if ST_BIFNodeATGF_MS_CUDA.cuda_source is not None:
            return  # 已经加载过

        # ── 1. 组合待检查的候选路径 ─────────────────────────────
        candidates = []

        # ① 相对于当前 .py 的固定位置
        here = Path(__file__).resolve().parent
        candidates.append(here / "cuda_snn_kernels.cu")
        candidates.append(here / "neuron_cupy" / "cuda_snn_kernels.cu")

        # ② 开发者可通过环境变量覆盖
        env_path = os.getenv("CUDA_SNN_KERNELS_PATH")
        if env_path:
            candidates.append(Path(env_path))

        # ③ 已打包的资源路径（importlib.resources 在 Py≥3.9 支持 pathlib API）
        try:
            resource_path = pkg_resources.files("neuron_cupy") / "cuda_snn_kernels.cu"
            candidates.append(resource_path)
        except ModuleNotFoundError:
            # 如果包本身不存在（纯源码工程时也可能发生），直接跳过
            pass

        # ── 2. 实际读取文件 ────────────────────────────────────
        cuda_code = None
        for p in candidates:
            try:
                if p.is_file():
                    cuda_code = p.read_text(encoding="utf-8")
                    break
            except Exception:
                # 有些路径（尤其是 importlib.resources 返回的）可能不支持 is_file(); 忽略即可
                continue

        if cuda_code is None:
            searched = "\n  - ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                "无法找到 cuda_snn_kernels.cu，已尝试：\n  - " + searched
            )

        ST_BIFNodeATGF_MS_CUDA.cuda_source = cuda_code

        # ── 3. 编译并缓存 CUDA RawModule ───────────────────────
        #    如果需要特殊编译选项，可在 options/disable_cooperative 等处调整
        module = cp.RawModule(code=cuda_code, options=("--std=c++14",))

        for precision in ("fp16", "fp32", "fp64"):
            setattr(
                ST_BIFNodeATGF_MS_CUDA,
                f"forward_kernel_{precision}",
                module.get_function(f"forward_kernel_{precision}"),
            )
            setattr(
                ST_BIFNodeATGF_MS_CUDA,
                f"backward_kernel_{precision}",
                module.get_function(f"backward_kernel_{precision}"),
            )

    # def _load_cuda_kernels():
    #     if ST_BIFNodeATGF_MS_CUDA.cuda_source is None:
    #         with open('./neuron_cupy/cuda_snn_kernels.cu', 'r') as f:
    #             ST_BIFNodeATGF_MS_CUDA.cuda_source = f.read()
                
    #         # Load CUDA kernels into CuPy
    #         module = cp.RawModule(code=ST_BIFNodeATGF_MS_CUDA.cuda_source)
    #         ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32 = module.get_function('forward_kernel_fp32')
    #         ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32 = module.get_function('backward_kernel_fp32')
    #         ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16 = module.get_function('forward_kernel_fp16')
    #         ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16 = module.get_function('backward_kernel_fp16')
    #         ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp64 = module.get_function('forward_kernel_fp64')
    #         ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp64 = module.get_function('backward_kernel_fp64')

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):
        ST_BIFNodeATGF_MS_CUDA._load_cuda_kernels()
        
        # Get dimensions
        time_steps, batch_size, *features = x_seq.shape
        features_flat = torch.prod(torch.tensor(features)).item()
        
        # Determine precision
        is_double = x_seq.dtype == torch.float64
        is_half = x_seq.dtype == torch.float16
        
        if is_double:
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp64
            v_th = v_th.type(torch.float64)
            T_max = T_max.type(torch.float64)
            T_min = T_min.type(torch.float64)
        elif is_half:
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16
            v_th = v_th.type(torch.float16)
            T_max = T_max.type(torch.float16)
            T_min = T_min.type(torch.float16)
        else:  # Default to fp32
            forward_kernel = ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32
        
        # Prepare output tensors with same dtype as input
        spike_seq_out = torch.zeros((time_steps + 1, batch_size, *features), 
                                  device=x_seq.device, dtype=x_seq.dtype)
        T_seq_out = torch.zeros_like(spike_seq_out)
        H_seq_out = torch.zeros_like(spike_seq_out)
        v_out = torch.zeros((batch_size, *features), 
                          device=x_seq.device, dtype=x_seq.dtype)
        
        # Convert tensors to CuPy
        x_seq_cp = cp.from_dlpack(to_dlpack(x_seq.contiguous()))
        v_th_cp = cp.from_dlpack(to_dlpack(v_th.contiguous()))
        T_max_cp = cp.from_dlpack(to_dlpack(T_max.contiguous()))
        T_min_cp = cp.from_dlpack(to_dlpack(T_min.contiguous()))
        prefire_cp = cp.from_dlpack(to_dlpack(prefire.contiguous()))
        spike_seq_cp = cp.from_dlpack(to_dlpack(spike_seq_out.contiguous()))
        v_out_cp = cp.from_dlpack(to_dlpack(v_out.contiguous()))
        T_seq_cp = cp.from_dlpack(to_dlpack(T_seq_out.contiguous()))
        H_seq_cp = cp.from_dlpack(to_dlpack(H_seq_out.contiguous()))
        
        # Launch kernel
        threads_per_block = 256
        blocks = (batch_size * features_flat + threads_per_block - 1) // threads_per_block
        
        forward_kernel(
            (blocks,), (threads_per_block,),
            (x_seq_cp, v_th_cp, T_max_cp, T_min_cp, prefire_cp,
             spike_seq_cp, v_out_cp, T_seq_cp, H_seq_cp,
             batch_size, time_steps, features_flat)
        )
        
        # Convert back to PyTorch - create fresh copies to avoid capsule reuse
        spike_seq = spike_seq_out.clone()
        v = v_out.clone()
        T_seq = T_seq_out.clone()
        H_seq = H_seq_out.clone()
        
        # print("ST_BIFNodeATGF_MS_CUDA:", spike_seq.dtype, H_seq.dtype, T_seq.dtype, x_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype)

        ctx.save_for_backward(spike_seq, T_seq, H_seq, v_th, T_max, T_min)
        ctx.is_double = is_double
        ctx.is_half = is_half
        
        return spike_seq[1:], v, T_seq[1:]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v: torch.Tensor, grad_T_seq: torch.Tensor):
        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        is_double = ctx.is_double
        is_half = ctx.is_half
        
        if is_double:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp64
        elif is_half:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16
        else:
            backward_kernel = ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32
        
        time_steps = grad_spike_seq.shape[0]
        batch_size = grad_spike_seq.shape[1]
        features_flat = grad_spike_seq[0].numel() // batch_size
        
        # Prepare output tensor
        grad_x_seq = torch.zeros_like(grad_spike_seq)
        
        # Convert tensors to CuPy
        grad_spike_seq_cp = cp.from_dlpack(to_dlpack(grad_spike_seq.contiguous()))
        grad_v_cp = cp.from_dlpack(to_dlpack(grad_v.contiguous()))
        grad_T_seq_cp = cp.from_dlpack(to_dlpack(grad_T_seq.contiguous()))
        spike_seq_cp = cp.from_dlpack(to_dlpack(spike_seq.contiguous()))
        T_seq_cp = cp.from_dlpack(to_dlpack(T_seq.contiguous()))
        H_seq_cp = cp.from_dlpack(to_dlpack(H_seq.contiguous()))
        v_th_cp = cp.from_dlpack(to_dlpack(v_th.contiguous()))
        T_max_cp = cp.from_dlpack(to_dlpack(T_max.contiguous()))
        T_min_cp = cp.from_dlpack(to_dlpack(T_min.contiguous()))
        grad_x_seq_cp = cp.from_dlpack(to_dlpack(grad_x_seq.contiguous()))
        
        # Launch kernel
        threads_per_block = 256
        blocks = (batch_size * features_flat + threads_per_block - 1) // threads_per_block
        
        backward_kernel(
            (blocks,), (threads_per_block,),
            (grad_spike_seq_cp, grad_v_cp, grad_T_seq_cp,
             spike_seq_cp, T_seq_cp, H_seq_cp,
             v_th_cp, T_max_cp, T_min_cp,
             grad_x_seq_cp,
             batch_size, time_steps, features_flat)
        )
        
        # Convert back to PyTorch - use original tensor to avoid capsule reuse
        grad_x = grad_x_seq.clone()
        
        # print("ST_BIFNodeATGF_MS_CUDA Backward:", grad_x_seq.dtype, grad_v_cp.dtype, grad_T_seq_cp.dtype, spike_seq_cp.dtype, spike_seq.dtype, H_seq.dtype, T_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype)

        # global first
        # if first:
        #     print('=================================================================')
        #     # print("v_th",v_th)
        #     # ST_BIFNodeATGF_MS_CUDA.v_thr1 = ST_BIFNodeATGF_MS_CUDA.v_thr1 + v_th
        #     # ST_BIFNodeATGF_MS_CUDA.count = ST_BIFNodeATGF_MS_CUDA.count + 1
        #     # print("vthr Sum",ST_BIFNodeATGF_MS_CUDA.v_thr1, "count", ST_BIFNodeATGF_MS_CUDA.count)
            
        #     for t in range(4):
        #         print("t=",t+1)
        #         print("H_seq_cp",H_seq[t+1].abs().mean())
        #         print("grad_x",grad_x[t].abs().mean())
        #         print("grad_spike_seq",grad_spike_seq[t].abs().mean())
        #         print("grad_x/grad_spike_seq",(torch.abs(grad_x)/(torch.abs(grad_spike_seq)+1e-5))[t].abs().mean())
        #     first = False 
            
        return grad_x, None, None, None, None
