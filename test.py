# test_snn.py

import torch
import pytest
from ifneuron_models import OriginalIFNeuron, OptimizedIFNeuron, CudaIFNeuron

def build_sparse_inputs(batch_size, num_features, timesteps=1, sparsity=0.0, device="cuda"):
    inputs_list = []
    for _ in range(timesteps):
        data = torch.randn(batch_size, num_features, dtype=torch.float32, device=device)
        if sparsity > 0:
            mask = torch.rand_like(data)
            data[mask < sparsity] = 0.0
        inputs_list.append(data)
    return inputs_list

def run_model_sequence_and_collect_states(model, inputs_list):
    """
    让模型在一串多步输入上运行，并在每个 timestep 后记录关键内部状态。
    返回 states: list of dict, 其中每个 dict 包含 {'q':..., 'acc_q':..., 'cur_output':...}
    """
    model.reset()
    states = []
    with torch.no_grad():
        for x_t in inputs_list:
            _ = model(x_t)
            # 收集当前时刻的内部状态
            # 这里要 clone 一下，避免后续 in-place 修改影响记录
            states.append({
                "q": model.q.clone(),
                "acc_q": model.acc_q.clone(),
                "cur_output": model.cur_output.clone()
            })
    return states

def assert_states_close(state_a, state_b, rtol=1e-5, atol=1e-7):
    """
    检查 state_a 和 state_b 的 q, acc_q, cur_output 三者在一定公差内是否相等。
    """
    for k in ["q", "acc_q", "cur_output"]:
        assert k in state_a and k in state_b, f"Missing key {k} in states."
        tensor_a = state_a[k]
        tensor_b = state_b[k]
        # 使用 torch.allclose 来比较
        if not torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol):
            max_diff = (tensor_a - tensor_b).abs().max().item()
            raise AssertionError(
                f"State key='{k}' not close. Max diff={max_diff:.2e}, "
                f"rtol={rtol}, atol={atol}"
            )

@pytest.mark.parametrize("ModelClass", [OptimizedIFNeuron, CudaIFNeuron])
@pytest.mark.parametrize("timesteps", [1, 16, 64])
@pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.5, 0.9])
def test_neuron_equivalence(ModelClass, timesteps, sparsity):
    """
    对比 OriginalIFNeuron 与 ModelClass (OptimizedIFNeuron 或 CudaIFNeuron)
    在多步输入下的内部状态是否一致（在一定容忍度内）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 神经元参数
    q_threshold = 0.01
    level = torch.tensor(16, dtype=torch.int32)
    sym = False

    # 实例化两个模型
    model_ref = OriginalIFNeuron(q_threshold, level, sym=sym).to(device)
    model_test = ModelClass(q_threshold, level, sym=sym).to(device)

    # 输入参数
    batch_size = 128
    num_features = 128

    # 构造多步稀疏输入
    inputs_seq = build_sparse_inputs(batch_size, num_features, timesteps=timesteps, sparsity=sparsity, device=device)

    # 跑一遍序列，记录每个模型在每个 timestep 的状态
    states_ref = run_model_sequence_and_collect_states(model_ref, inputs_seq)
    states_test = run_model_sequence_and_collect_states(model_test, inputs_seq)

    # 比较每个 timestep 下的状态
    for t in range(timesteps):
        assert_states_close(states_ref[t], states_test[t], rtol=1e-5, atol=1e-7)

    # 如果都通过，则认为两者等效
    print(f"{ModelClass.__name__} is equivalent to OriginalIFNeuron on {timesteps} timesteps input.")


if __name__ == "__main__":
    # 如果不用 pytest 命令行，可以直接在 main 中手动执行
    # 但推荐使用: pytest test_snn.py
    pytest.main([__file__])
