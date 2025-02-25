#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel 函数，简化实现：计算 tmp_spike, 更新 q 和 acc_q
__global__ void stbif_forward_kernel(
    const float *__restrict__ x,   // scaled input
    float *__restrict__ q,         // internal q
    float *__restrict__ acc_q,     // accumulator
    float *__restrict__ tmp_spike, // output spike
    const float pos_max,
    const float neg_min,
    const float q_threshold,
    const float eps,
    const int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
    {
        // 1. Add x to q
        q[idx] += x[idx];

        // 2. Round acc_q
        acc_q[idx] = roundf(acc_q[idx]);

        // 3. Build mask
        bool pos_mask = (q[idx] >= 1.0f) && (acc_q[idx] < pos_max);
        bool neg_mask = (q[idx] < -eps) && (acc_q[idx] > neg_min);

        // 4. Build spike
        float spike_val = 0.0f;
        if (pos_mask)
            spike_val = 1.0f;
        if (neg_mask)
            spike_val = -1.0f;
        tmp_spike[idx] = spike_val;

        // 5. Update q
        if (pos_mask)
            q[idx] -= 1.0f;
        if (neg_mask)
            q[idx] += 1.0f;

        // 6. Update acc_q
        acc_q[idx] += spike_val;

        // 到此，tmp_spike[idx] 还只是 +1/-1/0，还没乘以阈值
        // 如果要立刻返回真正输出，也可以这里做一次乘法
        tmp_spike[idx] *= q_threshold;
    }
}

std::vector<at::Tensor> stbif_forward_cuda(
    at::Tensor x,
    at::Tensor q,
    at::Tensor acc_q,
    float pos_max,
    float neg_min,
    float q_threshold,
    float eps)
{
    // 假设 x, q, acc_q 都是同样的 shape
    // 这里不做太多检查，实际需要做类型、设备、形状检查
    auto tmp_spike = torch::zeros_like(x);

    int numel = x.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    // 调用 Kernel
    stbif_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        q.data_ptr<float>(),
        acc_q.data_ptr<float>(),
        tmp_spike.data_ptr<float>(),
        pos_max,
        neg_min,
        q_threshold,
        eps,
        numel);

    // 同步一下，检查是否有 CUDA error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error in stbif_forward_kernel: %s\n", cudaGetErrorString(err));
        // 这里可以抛异常或者根据需要处理
    }
    return {tmp_spike, q, acc_q};
}

// PYBIND11 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stbif_forward", &stbif_forward_cuda, "STBIF forward (CUDA)");
}
