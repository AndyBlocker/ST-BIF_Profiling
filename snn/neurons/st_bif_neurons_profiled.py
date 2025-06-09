"""
ST-BIF (Spike Threshold - Bifurcation) Neuron Models with NVTX Profiling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron_cupy.cuda_operator import ST_BIFNodeATGF_MS_CUDA

# NVTX profiling support
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    # Fallback when NVTX is not available
    class nvtx:
        @staticmethod
        def range(name, color=None):
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()
    NVTX_AVAILABLE = False


def theta_backward(x):
    """Backward pass function for theta (step function) with sigmoid approximation"""
    with nvtx.range("theta_backward", color="blue"):
        sigmoid = torch.sigmoid(4*x)
        return 4*sigmoid*(1-sigmoid)


def theta(x):
    """Step function: returns 1 if x > 0, 0 otherwise"""
    with nvtx.range("theta_function", color="blue"):
        return 1.0*(torch.gt(x,0))
 

def theta_eq(x):
    """Step function: returns 1 if x >= 0, 0 otherwise"""
    with nvtx.range("theta_eq_function", color="blue"):
        return 1.0*(torch.ge(x,0))


class ST_BIFNodeATGF_SS(torch.autograd.Function):
    """Single-step ST-BIF Node Autograd Function with NVTX profiling"""
    
    @staticmethod
    def forward(ctx, x_t: torch.Tensor, V_t_1: torch.Tensor, T_t_1: torch.Tensor, v_th: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor, t: torch.Tensor):
        with nvtx.range("ST-BIF_SS_forward", color="red"):
            spike = x_t * 0.0
            H_t = V_t_1 + x_t
            
            with nvtx.range("spike_conditions", color="orange"):
                spike_condition = (H_t >= v_th) & (T_t_1-T_max < 0)
                neg_spike_condition = (H_t < 0) & (T_t_1-T_min > 0)
                
                spike = torch.where(spike_condition, torch.ones_like(H_t),
                                  torch.where(neg_spike_condition, -torch.ones_like(H_t),
                                            torch.zeros_like(H_t)))
            
            with nvtx.range("membrane_update", color="yellow"):
                V_t = H_t - spike * v_th
            
            with nvtx.range("timer_update", color="green"):
                T_t = (T_t_1 + 1) * (1 - torch.abs(spike)) + T_min * spike
            
            ctx.save_for_backward(x_t, V_t_1, T_t_1, v_th, T_max, T_min, t)
            return spike, V_t, T_t
    
    @staticmethod
    def backward(ctx, grad_spike, grad_V_t, grad_T_t):
        with nvtx.range("ST-BIF_SS_backward", color="purple"):
            x_t, V_t_1, T_t_1, v_th, T_max, T_min, t = ctx.saved_tensors
            
            with nvtx.range("gradient_computation", color="pink"):
                # Approximate gradient computation
                grad_x_t = grad_spike * theta_backward(V_t_1 + x_t - v_th)
                grad_V_t_1 = grad_spike * theta_backward(V_t_1 + x_t - v_th)
                grad_T_t_1 = None
                grad_v_th = -grad_spike * theta_backward(V_t_1 + x_t - v_th)
                grad_T_max = None
                grad_T_min = None
                grad_t = None
            
            return grad_x_t, grad_V_t_1, grad_T_t_1, grad_v_th, grad_T_max, grad_T_min, grad_t


class ST_BIFNodeATGF_MS(torch.autograd.Function):
    """Multi-step ST-BIF Node Autograd Function with NVTX profiling"""
    
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: torch.Tensor, 
                T_init: torch.Tensor, T_max: torch.Tensor, T_min: torch.Tensor):
        with nvtx.range("ST-BIF_MS_forward", color="red"):
            T = x_seq.shape[0]
            device = x_seq.device
            
            # Initialize variables
            with nvtx.range("initialization", color="cyan"):
                v = v_init.clone()
                T_t = T_init.clone()
                spike_seq = torch.zeros_like(x_seq)
                V_seq = torch.zeros_like(x_seq)
                T_seq = torch.zeros_like(x_seq)
            
            # Process each time step
            for t in range(T):
                with nvtx.range(f"timestep_{t}", color="lime"):
                    x_t = x_seq[t]
                    
                    with nvtx.range("membrane_potential", color="orange"):
                        H_t = v + x_t
                    
                    with nvtx.range("spike_decision", color="yellow"):
                        spike_condition = (H_t >= v_th) & (T_t < T_max)
                        neg_spike_condition = (H_t < 0) & (T_t > T_min)
                        
                        spike = torch.where(spike_condition, torch.ones_like(H_t),
                                          torch.where(neg_spike_condition, -torch.ones_like(H_t),
                                                    torch.zeros_like(H_t)))
                    
                    with nvtx.range("state_update", color="green"):
                        v = H_t - spike * v_th
                        T_t = (T_t + 1) * (1 - torch.abs(spike)) + T_min * spike
                    
                    with nvtx.range("store_results", color="blue"):
                        spike_seq[t] = spike
                        V_seq[t] = v
                        T_seq[t] = T_t
            
            ctx.save_for_backward(x_seq, v_init, v_th, T_init, T_max, T_min, spike_seq, V_seq, T_seq)
            return spike_seq, V_seq, T_seq
    
    @staticmethod
    def backward(ctx, grad_spike_seq, grad_V_seq, grad_T_seq):
        with nvtx.range("ST-BIF_MS_backward", color="purple"):
            x_seq, v_init, v_th, T_init, T_max, T_min, spike_seq, V_seq, T_seq = ctx.saved_tensors
            T = x_seq.shape[0]
            
            with nvtx.range("gradient_initialization", color="pink"):
                grad_x_seq = torch.zeros_like(x_seq)
                grad_v_init = torch.zeros_like(v_init)
                grad_v_th = torch.zeros_like(v_th)
                grad_T_init = None
                grad_T_max = None
                grad_T_min = None
            
            # Backward through time
            for t in reversed(range(T)):
                with nvtx.range(f"backward_timestep_{t}", color="magenta"):
                    # Simplified gradient computation
                    if t == 0:
                        v_prev = v_init
                    else:
                        v_prev = V_seq[t-1]
                    
                    H_t = v_prev + x_seq[t]
                    grad_spike_t = grad_spike_seq[t]
                    
                    # Approximate gradients
                    grad_x_seq[t] = grad_spike_t * theta_backward(H_t - v_th)
                    grad_v_th += -grad_spike_t * theta_backward(H_t - v_th)
                    
                    if t == 0:
                        grad_v_init += grad_spike_t * theta_backward(H_t - v_th)
            
            return grad_x_seq, grad_v_init, grad_v_th, grad_T_init, grad_T_max, grad_T_min


class ST_BIFNeuron_SS(nn.Module):
    """Single-step ST-BIF Neuron with NVTX profiling"""
    
    def __init__(self, q_threshold=1.0, level=8, learnable=True):
        super(ST_BIFNeuron_SS, self).__init__()
        
        with nvtx.range("ST-BIF_SS_init", color="gray"):
            self.q_threshold = nn.Parameter(torch.tensor(q_threshold), requires_grad=learnable)
            self.level = torch.tensor(level)
            self.register_buffer("T_max", torch.tensor(level))
            self.register_buffer("T_min", torch.tensor(-level))
            self.register_buffer("pos_max", torch.tensor(level - 1))
            
            # Initialize state variables
            self.V = None
            self.T = None
            self.step = 0
    
    def forward(self, x):
        with nvtx.range("ST-BIF_SS_forward_call", color="red"):
            if self.V is None:
                with nvtx.range("state_initialization", color="cyan"):
                    self.V = torch.zeros_like(x)
                    self.T = torch.zeros_like(x)
            
            spike, self.V, self.T = ST_BIFNodeATGF_SS.apply(
                x, self.V, self.T, self.q_threshold, self.T_max, self.T_min, self.step
            )
            
            self.step += 1
            return spike
    
    def reset(self):
        with nvtx.range("ST-BIF_SS_reset", color="gray"):
            self.V = None
            self.T = None
            self.step = 0


class ST_BIFNeuron_MS(nn.Module):
    """Multi-step ST-BIF Neuron with NVTX profiling"""
    
    def __init__(self, q_threshold=1.0, level=8, learnable=True):
        super(ST_BIFNeuron_MS, self).__init__()
        
        with nvtx.range("ST-BIF_MS_init", color="gray"):
            self.q_threshold = nn.Parameter(torch.tensor(q_threshold), requires_grad=learnable)
            self.level = torch.tensor(level)
            self.register_buffer("T_max", torch.tensor(level))
            self.register_buffer("T_min", torch.tensor(-level))
            self.register_buffer("pos_max", torch.tensor(level - 1))
            
            # Multi-step processing parameters
            self.T = 8  # Default time steps
            self.step = 0
    
    def forward(self, x):
        with nvtx.range("ST-BIF_MS_forward_call", color="red"):
            # Handle multi-step input format
            if x.dim() == 4:  # [B, C, H, W] -> expand to [T, B, C, H, W]
                with nvtx.range("input_expansion", color="cyan"):
                    x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            elif x.dim() == 5:  # Already in [T, B, C, H, W] format
                pass
            else:
                raise ValueError(f"Unexpected input dimension: {x.dim()}")
            
            T, B = x.shape[0], x.shape[1]
            
            # Initialize states
            with nvtx.range("state_initialization", color="cyan"):
                v_init = torch.zeros(B, *x.shape[2:], device=x.device)
                T_init = torch.zeros_like(v_init)
            
            # Process through time steps
            spike_seq, V_seq, T_seq = ST_BIFNodeATGF_MS.apply(
                x, v_init, self.q_threshold, T_init, self.T_max, self.T_min
            )
            
            with nvtx.range("output_aggregation", color="lime"):
                # Average spikes over time steps for final output
                output = spike_seq.mean(dim=0)
            
            self.step += 1
            return output
    
    def reset(self):
        with nvtx.range("ST-BIF_MS_reset", color="gray"):
            self.step = 0


class ST_BIFNeuron_MS_CUDA(nn.Module):
    """CUDA-accelerated Multi-step ST-BIF Neuron with NVTX profiling"""
    
    def __init__(self, q_threshold=1.0, level=8, learnable=True):
        super(ST_BIFNeuron_MS_CUDA, self).__init__()
        
        with nvtx.range("ST-BIF_MS_CUDA_init", color="gray"):
            self.q_threshold = nn.Parameter(torch.tensor(q_threshold), requires_grad=learnable)
            self.level = torch.tensor(level)
            self.register_buffer("T_max", torch.tensor(level))
            self.register_buffer("T_min", torch.tensor(-level))
            self.register_buffer("pos_max", torch.tensor(level - 1))
            
            self.T = 8
            self.step = 0
    
    def forward(self, x):
        with nvtx.range("ST-BIF_MS_CUDA_forward", color="red"):
            # Use CUDA kernel for acceleration
            if x.dim() == 4:
                with nvtx.range("cuda_input_prep", color="cyan"):
                    x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            
            T, B = x.shape[0], x.shape[1]
            
            with nvtx.range("cuda_state_init", color="cyan"):
                v_init = torch.zeros(B, *x.shape[2:], device=x.device)
                T_init = torch.zeros_like(v_init)
            
            with nvtx.range("cuda_kernel_call", color="orange"):
                # Call optimized CUDA kernel
                spike_seq, V_seq, T_seq = ST_BIFNodeATGF_MS_CUDA.apply(
                    x, v_init, self.q_threshold, T_init, self.T_max, self.T_min
                )
            
            with nvtx.range("cuda_output_process", color="lime"):
                output = spike_seq.mean(dim=0)
            
            self.step += 1
            return output
    
    def reset(self):
        with nvtx.range("ST-BIF_MS_CUDA_reset", color="gray"):
            self.step = 0


# Factory function for creating profiled ST-BIF neurons
def create_st_bif_neuron(neuron_type="MS", q_threshold=1.0, level=8, learnable=True, use_cuda=False):
    """Factory function to create ST-BIF neurons with profiling"""
    with nvtx.range("create_st_bif_neuron", color="gray"):
        if neuron_type == "SS":
            return ST_BIFNeuron_SS(q_threshold, level, learnable)
        elif neuron_type == "MS":
            if use_cuda and torch.cuda.is_available():
                return ST_BIFNeuron_MS_CUDA(q_threshold, level, learnable)
            else:
                return ST_BIFNeuron_MS(q_threshold, level, learnable)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")