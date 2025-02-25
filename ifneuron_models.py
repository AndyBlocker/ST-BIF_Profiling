import torch
import torch.nn as nn
from torch.cuda import nvtx
import stbif_cuda


class OriginalIFNeuron(nn.Module):
    def __init__(self, q_threshold, level, sym=False):
        super(OriginalIFNeuron, self).__init__()
        self.q_threshold = q_threshold
        
        self.q = 0.0
        self.acc_q = 0.0
        self.cur_output = 0.0
        
        self.is_work = False
        self.level = level.clone().detach()
        self.sym = sym
        
        if sym:
            self.pos_max = torch.tensor(self.level // 2 - 1)
            self.neg_min = torch.tensor(-self.level // 2)
        else:
            self.pos_max = (self.level - 1).clone().detach()
            self.neg_min = torch.tensor(0)
        
        self.eps = 0

    def reset(self):
        self.q = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False

    def forward(self, input):
        with nvtx.range("IFNeuron forward"):
            # -------------------------
            # Input Scaling
            # -------------------------
            with nvtx.range("Input Scaling"):
                x = input / self.q_threshold
            
            # -------------------------
            # First Working Judging
            # -------------------------
            with nvtx.range("First Working Judging"):
                if (not torch.is_tensor(x)) and x == 0.0 \
                   and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
                    self.is_work = False
                    return x

            # -------------------------
            # Allocating for first input tensor
            # -------------------------
            with nvtx.range("Allocating for first input tensor"):
                if not torch.is_tensor(self.cur_output):
                    self.cur_output = torch.zeros_like(x)
                if not torch.is_tensor(self.acc_q):
                    self.acc_q = torch.zeros_like(x)
                if not torch.is_tensor(self.q):
                    self.q = torch.zeros_like(x) + 0.5

            self.is_work = True
            
            # -------------------------
            # Add Q
            # -------------------------
            with nvtx.range("Add Q"):
                self.q = self.q + (x.detach() if torch.is_tensor(x) else x)

            # -------------------------
            # Round Q
            # -------------------------
            with nvtx.range("Round Q"):
                self.acc_q = torch.round(self.acc_q)

            # -------------------------
            # Build Spike Mask
            # -------------------------
            with nvtx.range("Build Spike Mask"):
                with nvtx.range("pos_mask"):
                    spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max)
                with nvtx.range("neg_mask"):
                    neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_min)

            # -------------------------
            # Build Output
            # -------------------------
            with nvtx.range("Build Output"):
                self.cur_output[:] = 0
                self.cur_output[spike_position] = 1
                self.cur_output[neg_spike_position] = -1
            
            # -------------------------
            # Update Q
            # -------------------------
            # with nvtx.range("Update Q"):
                self.q[spike_position] -= 1
                self.q[neg_spike_position] += 1

            # -------------------------
            # Update Spike Tracer
            # -------------------------
            with nvtx.range("Update Q, Spike Tracer and cur_output"):
                # self.q = self.q + self.cur_output
                self.acc_q = self.acc_q + self.cur_output
                self.cur_output = self.cur_output * self.q_threshold

            # -------------------------
            # Check If Working
            # -------------------------
            with nvtx.range("Check If Working"):
                if (x == 0).all() and (self.cur_output == 0).all():
                    self.is_work = False
            
            return self.cur_output


class OptimizedIFNeuron(nn.Module):
    def __init__(self, q_threshold, level, sym=False):
        super(OptimizedIFNeuron, self).__init__()
        
        self.q = 0.0
        self.acc_q = 0.0
        self.cur_output = 0.0
        
        self.q_threshold = q_threshold
        self.is_work = False
        
        self.level = level.clone().detach()
        self.sym = sym
        if sym:
            self.pos_max = torch.tensor(self.level // 2 - 1)
            self.neg_min = torch.tensor(-self.level // 2)
        else:
            self.pos_max = (self.level - 1).clone().detach()
            self.neg_min = torch.tensor(0)

        self.eps = 0

    def reset(self):
        self.q = 0.0
        self.acc_q = 0.0
        self.cur_output = 0.0
        self.is_work = False

    def forward(self, input):
        with nvtx.range("IFNeuron forward"):
            with nvtx.range("Input Scaling"):
                x = input / self.q_threshold
            
            # not working
            with nvtx.range("First Working Judging"):
                if isinstance(x, (float, int)) and x == 0.0 \
                and isinstance(self.cur_output, (float, int)) and self.cur_output == 0.0:
                    self.is_work = False
                    return x
            
            with nvtx.range("Allocating for first input tensor"):
                if isinstance(self.cur_output, (float, int)):
                    self.cur_output = torch.zeros_like(x)
                if isinstance(self.acc_q, (float, int)):
                    self.acc_q = torch.zeros_like(x)
                if isinstance(self.q, (float, int)):
                    self.q = torch.full_like(x, 0.5)
            
            self.is_work = True
            
            with nvtx.range("Add Q"):
                self.q.add_(x.detach() if torch.is_tensor(x) else x)

            with nvtx.range("Round Q"):
                self.acc_q.round_()
            
            with nvtx.range("Build Spike Mask"):
                with nvtx.range("pos_mask"):
                    pos_mask = (self.q >= 1.0) & (self.acc_q < self.pos_max) 
                with nvtx.range("neg_mask"):    
                    neg_mask = (self.q < -self.eps) & (self.acc_q > self.neg_min) 

                with nvtx.range("pos_val"):
                    pos_val = pos_mask.to(self.q.dtype)
                with nvtx.range("neg_val"):
                    neg_val = neg_mask.to(self.q.dtype)
                with nvtx.range("build output"):
                    tmp_spike = pos_val - neg_val
            
            with nvtx.range("update Q"):
                self.q.sub_(pos_val).add_(neg_val)

            with nvtx.range("update Spike Tracer"):
                self.acc_q.add_(tmp_spike)

            with nvtx.range("Build Output"):
                self.cur_output = tmp_spike.mul(self.q_threshold)

            with nvtx.range("Check If Working"):
                if x.abs().sum() == 0 and tmp_spike.abs().sum() == 0:
                    self.is_work = False

            return self.cur_output


class CudaIFNeuron(nn.Module):
    """
    使用自定义 CUDA Kernel 的 IF Neuron。
    """
    def __init__(self, q_threshold, level, sym=False):
        super(CudaIFNeuron, self).__init__()
        self.q_threshold = q_threshold

        self.q = 0.0
        self.acc_q = 0.0
        self.cur_output = 0.0

        self.is_work = False
        self.level = level.clone().detach()
        self.sym = sym

        if sym:
            self.pos_max = torch.tensor(self.level // 2 - 1)
            self.neg_min = torch.tensor(-self.level // 2)
        else:
            self.pos_max = (self.level - 1).clone().detach()
            self.neg_min = torch.tensor(0)

        self.eps = 0

    def reset(self):
        self.q = 0.0
        self.acc_q = 0.0
        self.cur_output = 0.0
        self.is_work = False

    def forward(self, input):
        with nvtx.range("CudaIFNeuron forward"):
            with nvtx.range("Input Scaling"):
                x = input / self.q_threshold

            # 如果还是 float/int, 需要初始化为张量
            with nvtx.range("Allocating Tensors"):
                if isinstance(self.cur_output, (float, int)):
                    self.cur_output = torch.zeros_like(x)
                if isinstance(self.acc_q, (float, int)):
                    self.acc_q = torch.zeros_like(x)
                if isinstance(self.q, (float, int)):
                    # 比如初始 0.5
                    self.q = torch.full_like(x, 0.5)

            # 调用自定义 CUDA kernel
            with nvtx.range("CUDA Kernel: stbif_forward"):
                # stbif_forward 返回 [tmp_spike, q_new, acc_q_new]
                tmp_spike, q_new, acc_q_new = stbif_cuda.stbif_forward(
                    x,                # scaled input
                    self.q,           # internal q
                    self.acc_q,       # accumulator
                    float(self.pos_max.item()),
                    float(self.neg_min.item()),
                    float(self.q_threshold),
                    float(self.eps)
                )

            self.q = q_new
            self.acc_q = acc_q_new
            self.cur_output = tmp_spike  # tmp_spike 已经乘过阈值

            # 是否工作判断
            with nvtx.range("Check If Working"):
                if x.abs().sum() == 0 and tmp_spike.abs().sum() == 0:
                    self.is_work = False
                else:
                    self.is_work = True

            return self.cur_output
