import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack


first = True
class ST_BIFNodeATGF_MS_CUDA(torch.autograd.Function):
    cuda_source = None  # Will store the CUDA source code
    count = 0
    v_thr1 = 0
    
    @staticmethod
    def _load_cuda_kernels():
        if ST_BIFNodeATGF_MS_CUDA.cuda_source is None:
            with open('./neuron_cupy/cuda_snn_kernels.cu', 'r') as f:
                ST_BIFNodeATGF_MS_CUDA.cuda_source = f.read()
                
            # Load CUDA kernels into CuPy
            module = cp.RawModule(code=ST_BIFNodeATGF_MS_CUDA.cuda_source)
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp32 = module.get_function('forward_kernel_fp32')
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp32 = module.get_function('backward_kernel_fp32')
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp16 = module.get_function('forward_kernel_fp16')
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp16 = module.get_function('backward_kernel_fp16')
            ST_BIFNodeATGF_MS_CUDA.forward_kernel_fp64 = module.get_function('forward_kernel_fp64')
            ST_BIFNodeATGF_MS_CUDA.backward_kernel_fp64 = module.get_function('backward_kernel_fp64')

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