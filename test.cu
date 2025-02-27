#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include <nvtx3/nvToolsExt.h>

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
        // acc_q[idx] = roundf(acc_q[idx]);

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

        // 7. Multiply by threshold
        tmp_spike[idx] *= q_threshold;
    }
}

void stbif_forward_cuda(
    const float* x_d,
    float* q_d,
    float* acc_q_d,
    float* spike_d,
    float pos_max,
    float neg_min,
    float q_threshold,
    float eps,
    int numel,
    cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // kernel launch
    stbif_forward_kernel<<<blocks, threads, 0, stream>>>(
        x_d, q_d, acc_q_d, spike_d,
        pos_max, neg_min, q_threshold, eps, numel
    );

    // optional: check launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)
                  << " at stbif_forward_kernel launch." << std::endl;
    }
}

int main(int argc, char** argv)
{
    // ----------------- 参数 -----------------
    int batch_size = 128;
    int num_features = 128 * 16;
    int warmup_steps = 5;
    int profile_steps = 10;
    
    float q_threshold = 0.01f;
    float pos_max     = 15.f;  // 对应 level=16, sym=False => pos_max=level-1=15
    float neg_min     = 0.f;   
    float eps         = 0.f; 

    std::cout << "[INFO] batch_size=" << batch_size
              << ", num_features=" << num_features << std::endl;
    std::cout << "[INFO] warmup_steps=" << warmup_steps
              << ", profile_steps=" << profile_steps << std::endl;

    int numel = batch_size * num_features;

    std::vector<float> x_h(numel);
    std::vector<float> q_h(numel, 0.5f);     
    std::vector<float> acc_q_h(numel, 0.f);  
    std::vector<float> spike_h(numel, 0.f);  

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < numel; i++) {
        x_h[i] = dist(rng);
    }

    for (int i = 0; i < numel; i++) {
        x_h[i] /= q_threshold; 
    }

    float* x_d = nullptr;
    float* q_d = nullptr;
    float* acc_q_d = nullptr;
    float* spike_d = nullptr;

    cudaMalloc((void**)&x_d,     numel * sizeof(float));
    cudaMalloc((void**)&q_d,     numel * sizeof(float));
    cudaMalloc((void**)&acc_q_d, numel * sizeof(float));
    cudaMalloc((void**)&spike_d, numel * sizeof(float));

    // 拷贝 Host -> Device
    cudaMemcpy(x_d,     x_h.data(),     numel*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(q_d,     q_h.data(),     numel*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(acc_q_d, acc_q_h.data(), numel*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spike_d, spike_h.data(), numel*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    
    // ----------------- Warmup 阶段 -----------------
    std::cout << "[INFO] Warmup for " << warmup_steps << " steps...\n";
    for (int i = 0; i < warmup_steps; i++)
    {
        stbif_forward_cuda(
            x_d, q_d, acc_q_d, spike_d,
            pos_max, neg_min, q_threshold, eps,
            numel
        );
        cudaDeviceSynchronize();
    }
    std::cout << "[INFO] Warmup done.\n\n";

    // ----------------- Profile 阶段 + NVTX 标记 -----------------
    std::cout << "[INFO] Now running " << profile_steps << " steps for NCU profiling...\n";

    // 整个profile用一个大的 Range
    nvtxRangePushA("TotalProfileRegion");
    for (int step = 0; step < profile_steps; step++)
    {
        // 给每一步一个 Range
        std::string step_name = "Inference-Step-" + std::to_string(step);
        nvtxRangePushA(step_name.c_str());

        // 调 kernel
        stbif_forward_cuda(
            x_d, q_d, acc_q_d, spike_d,
            pos_max, neg_min, q_threshold, eps,
            numel
        );
        cudaDeviceSynchronize();

        // 模拟 "model.reset()" => 重置 q=0.5, acc_q=0, spike=0
        // 这里为了演示，与 Python reset() 类似
        cudaMemcpy(q_d,     q_h.data(),     numel*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(acc_q_d, acc_q_h.data(), numel*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(spike_d, spike_h.data(), numel*sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        nvtxRangePop(); // end of this step
    }
    nvtxRangePop(); // end of "TotalProfileRegion"

    std::cout << "[INFO] Finished profile steps.\n";

    // ----------------- 释放  -----------------
    cudaFree(x_d);
    cudaFree(q_d);
    cudaFree(acc_q_d);
    cudaFree(spike_d);

    std::cout << "[INFO] Script completed. You can now use Nsight Compute to analyze.\n";
    return 0;
}
