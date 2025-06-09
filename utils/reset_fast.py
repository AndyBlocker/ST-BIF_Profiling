import torch
from torch import nn

def fast_zero_states(module: nn.Module):
    """
    仅对**状态张量**做 in-place zero_() ，
    不重新分配，不动 weight / bias / running_mean …
    可安全用于 CUDA-Graph。
    """
    # 1. layer 自己实现了 zero_state() 时直接调用（最保险）
    if hasattr(module, 'zero_state'):
        module.zero_state()
        return

    # 2. 通用兜底：把非 Parameter/Buffer 的 float Tensor 属性全部 zero_
    for attr, val in module.__dict__.items():
        if not isinstance(val, torch.Tensor):
            continue                      # 不是张量 → 跳过
        if attr in module._parameters or attr in module._buffers:
            continue                      # weight / running_mean … → 跳过
        if val.is_floating_point():
            val.zero_()

    # 3. 递归子模块
    for child in module.children():
        fast_zero_states(child)
