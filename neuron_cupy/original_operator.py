import torch

def theta_backward(x, V_thr):
    """Gaussian approximation of surrogate gradient (handles tensor or scalar V_thr)."""
    if not torch.is_tensor(V_thr):
        V_thr = torch.as_tensor(V_thr, dtype=x.dtype, device=x.device)
    sigma = 0.405 * V_thr
    a = torch.reciprocal(V_thr)
    upper_x = -(x * x) / (2.0 * sigma * sigma)
    return a * torch.exp(upper_x)
#     # tanh = F.tanh(2*x)
#     # return 1/(1+(2*x)*(2*x))
#     # return 1 - F.tanh(2*x)*F.tanh(2*x)
# def theta_backward(x, V_thr):
#     mu = 0
#     sigma = 0.4*V_thr
#     a = 1.0/V_thr
#     upper_x = (-(x-mu)**2)/(2*sigma**2)
#     x = a * torch.exp(upper_x)
#     return x

# def theta_backward(x, V_thr):
#     A = 3.5
#     B = 3
#     x = A * torch.exp(-B*torch.abs(x)**(1/3))
#     return x

def theta(x):
    # return (x > 0).int()
    return (1.0*(torch.gt(x,0)))
 
def theta_eq(x):
    # return (x >= 0).int()
    return (1.0*(torch.ge(x,0)))

class ST_BIFNodeATGF_MS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, prefire: torch.Tensor):

        Time = x_seq.shape[0]
        orig_dtype = x_seq.dtype
        compute_dtype = torch.float32 if orig_dtype == torch.float16 else orig_dtype

        x_seq_c = x_seq.to(compute_dtype)
        v_th_c = v_th.to(compute_dtype)
        T_max_c = T_max.to(compute_dtype)
        T_min_c = T_min.to(compute_dtype)
        prefire_c = prefire.to(compute_dtype)

        spike_seq = []
        T_seq = []
        H_seq = []

        v = x_seq_c[0] * 0 + (0.5 + prefire_c) * v_th_c
        T = x_seq_c[0] * 0

        spike_init = torch.zeros_like(v)
        spike_seq.append(spike_init)
        T_seq.append(T)
        H_seq.append(v)

        tmax_val = float(T_max_c.item()) if T_max_c.numel() == 1 else None
        if tmax_val is not None:
            max_pf_steps = max(int(tmax_val), 0)
            max_pf_steps = min(max_pf_steps, Time)
            if tmax_val > 0:
                pf_delta = prefire_c * v_th_c / T_max_c
            else:
                pf_delta = torch.zeros((), dtype=compute_dtype, device=x_seq.device)
        else:
            raise ValueError("ST_BIFNodeATGF_MS expects scalar T_max for parity with CUDA kernel")

        one = torch.ones_like(v)

        for t in range(Time):
            v = v + x_seq_c[t]
            H_seq.append(v)

            cond_pos = torch.logical_and(torch.ge(v - v_th_c, 0), torch.lt(T - T_max_c, 0))
            cond_neg = torch.logical_and(torch.lt(v, 0), torch.gt(T - T_min_c, 0))

            spike = torch.zeros_like(v)
            spike = torch.where(cond_pos, one, spike)
            spike = torch.where(cond_neg, -one, spike)

            if t < max_pf_steps:
                v = v - v_th_c * spike - pf_delta
            else:
                v = v - v_th_c * spike

            T = T + spike
            T_seq.append(T)
            spike_seq.append(spike)

        H_seq = torch.stack(H_seq, dim=0)
        T_seq = torch.stack(T_seq, dim=0)
        spike_seq = torch.stack(spike_seq, dim=0)

        if compute_dtype != orig_dtype:
            H_seq = H_seq.to(orig_dtype)
            T_seq = T_seq.to(orig_dtype)
            spike_seq = spike_seq.to(orig_dtype)
            v = v.to(orig_dtype)

        ctx.save_for_backward(spike_seq, T_seq, H_seq, v_th, T_max, T_min)

        return spike_seq[1:,], v, T_seq[1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor, grad_T_seq: torch.Tensor):
        
        spike_seq, T_seq, H_seq, v_th, T_max, T_min = ctx.saved_tensors
        Time = spike_seq.shape[0] - 1

        orig_dtype = spike_seq.dtype
        compute_dtype = torch.float32 if orig_dtype == torch.float16 else orig_dtype

        # Promote to higher precision for stable comparison with CUDA kernels
        spike_seq_c = spike_seq.to(compute_dtype)
        T_seq_c = T_seq.to(compute_dtype)
        H_seq_c = H_seq.to(compute_dtype)
        v_th_c = v_th.to(compute_dtype)
        T_max_c = T_max.to(compute_dtype)
        T_min_c = T_min.to(compute_dtype)
        grad_Y_seq = grad_spike_seq.to(compute_dtype)

        grad_x_seq = []
        if grad_v_seq is None:
            grad_V = torch.zeros_like(grad_Y_seq[0])
        else:
            grad_V = grad_v_seq.to(compute_dtype)
        grad_T = torch.zeros_like(grad_Y_seq[0])
        one_tensor = torch.tensor(1.0, dtype=compute_dtype, device=H_seq_c.device)

        for t in range(Time, 0, -1):
            H_curr = H_seq_c[t]
            T_prev = T_seq_c[t-1]

            grad_T_t_to_H_t = (
                theta_backward(H_curr - v_th_c, v_th_c) * theta(T_max_c - T_prev)
                + theta_backward(-H_curr, v_th_c) * theta(T_prev - T_min_c)
            )
            grad_Y_t_to_T_t_1 = -(
                theta_eq(H_curr - v_th_c) * theta_backward(T_max_c - T_prev, one_tensor)
                + theta(-H_curr) * theta_backward(T_prev - T_min_c, one_tensor)
            )

            common = grad_Y_seq[t-1] - v_th_c * grad_V + grad_T
            grad_X = common * grad_T_t_to_H_t + grad_V
            grad_T = common * grad_Y_t_to_T_t_1 + grad_T
            grad_V = grad_X + 0.0
            grad_x_seq.append(grad_X)
        grad_x_seq = torch.flip(torch.stack(grad_x_seq,dim=0),dims=[0])
        if grad_x_seq.dtype != orig_dtype:
            grad_x_seq = grad_x_seq.to(orig_dtype)
        # print(spike_seq.dtype, T_seq.dtype, H_seq.dtype, v_th.dtype, T_max.dtype, T_min.dtype, grad_spike_seq.dtype, grad_x_seq.dtype, grad_v_seq.dtype, grad_T_seq.dtype)
        return grad_x_seq, None, None, None, None
