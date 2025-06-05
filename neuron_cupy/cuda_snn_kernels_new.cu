#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 使用常量内存存储全局参数，减少内存带宽需求
__constant__ float d_v_th_fp32;
__constant__ float d_T_max_fp32;
__constant__ float d_T_min_fp32;
__constant__ float d_prefire_fp32;

__constant__ __half d_v_th_fp16;
__constant__ __half d_T_max_fp16;
__constant__ __half d_T_min_fp16;
__constant__ __half d_prefire_fp16;

__constant__ double d_v_th_fp64;
__constant__ double d_T_max_fp64;
__constant__ double d_T_min_fp64;
__constant__ double d_prefire_fp64;

template<typename T>
__device__ __forceinline__ T theta(T x) {
    return x > T(0) ? T(1) : T(0);
}

template<typename T>
__device__ __forceinline__ T theta_eq(T x) {
    return x >= T(0) ? T(1) : T(0);
}

extern "C" {


__device__ __forceinline__ float theta_backward_fp32(float x, float V_thr, float S, float S_min, float S_max) {
    const float mu = 0.0f;
    const float sigma = 0.405f * V_thr;
    const float a = __fdividef(1.0f, V_thr); 
    const float diff = x - mu;
    const float upper_x = -diff * diff * __fdividef(0.5f, (sigma * sigma));
    return a * __expf(upper_x);  
}

__device__ __forceinline__ __half theta_backward_fp16(__half x, __half V_thr, __half S, __half S_min, __half S_max) {
    const __half mu = __float2half(0.0f);
    const __half sigma = __hmul(__float2half(0.405f), V_thr);
    const __half a = __hdiv(__float2half(1.0f), V_thr);
    const __half diff = __hsub(x, mu);
    const __half upper_x = __hdiv(__hmul(__hneg(diff), diff), __hmul(__float2half(2.0f), __hmul(sigma, sigma)));
    return __hmul(a, hexp(upper_x));
}

__device__ __forceinline__ double theta_backward_fp64(double x, double V_thr, double S, double S_min, double S_max) {
    const double mu = 0.0;
    const double sigma = 0.405 * V_thr;
    const double a = 1.0/V_thr;
    const double diff = x - mu;
    const double upper_x = -(diff * diff) / (2.0 * sigma * sigma);
    return a * exp(upper_x);
}

__global__ void __launch_bounds__(128, 8) forward_kernel_fp32(
    const float* __restrict__ x_seq,
    const float* v_th,
    const float* T_max,
    const float* T_min,
    const float* prefire,
    float* __restrict__ spike_seq_out,
    float* __restrict__ v_out,
    float* __restrict__ T_seq_out,
    float* __restrict__ H_seq_out,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ float shared_data[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_data[0] = v_th[0];
        shared_data[1] = T_max[0];
        shared_data[2] = T_min[0];
        shared_data[3] = prefire[0];
    }
    __syncthreads();
    
    const float local_v_th = shared_data[0];
    const float local_T_max = shared_data[1];
    const float local_T_min = shared_data[2];
    const float local_prefire = shared_data[3];
    
    float v = (0.5f + local_prefire) * local_v_th;
    float T = 0.0f;
    
    T_seq_out[idx] = T;
    spike_seq_out[idx] = 0.0f;
    H_seq_out[idx] = v;
    
    const int batch_stride = batch_size * features;
    
    #pragma unroll 4 
    for (int t = 0; t < time_steps; t++) {
        const int current_idx = t * batch_stride + idx;
        const int next_idx = (t + 1) * batch_stride + idx;
        
        v += x_seq[current_idx];
        H_seq_out[next_idx] = v;
        
        float spike = 0.0f;
        const bool pos_spike = (v >= local_v_th) & (T < local_T_max);
        const bool neg_spike = (v < 0.0f) & (T > local_T_min);
        
        spike = pos_spike ? 1.0f : (neg_spike ? -1.0f : 0.0f);
        
        const float v_reset = local_v_th * spike;
        const float prefire_reset = (t < local_T_max) ? 
            local_prefire * local_v_th / local_T_max : 0.0f;
        
        v -= (v_reset + prefire_reset);
        T += spike;
        
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    v_out[idx] = v;
}

__global__ void __launch_bounds__(128, 8) forward_kernel_fp16(
    const __half* __restrict__ x_seq,
    const __half* v_th,
    const __half* T_max,
    const __half* T_min,
    const __half* prefire,
    __half* __restrict__ spike_seq_out,
    __half* __restrict__ v_out,
    __half* __restrict__ T_seq_out,
    __half* __restrict__ H_seq_out,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ __half shared_data_half[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_data_half[0] = v_th[0];
        shared_data_half[1] = T_max[0];
        shared_data_half[2] = T_min[0];
        shared_data_half[3] = prefire[0];
    }
    __syncthreads();
    
    const __half local_v_th = shared_data_half[0];
    const __half local_T_max = shared_data_half[1];
    const __half local_T_min = shared_data_half[2];
    const __half local_prefire = shared_data_half[3];
    
    __half v = __hmul(__hadd(__float2half(0.5f), local_prefire), local_v_th);
    __half T = __float2half(0.0f);
    
    T_seq_out[idx] = T;
    spike_seq_out[idx] = __float2half(0.0f);
    H_seq_out[idx] = v;
    
    const int batch_stride = batch_size * features;
    
    #pragma unroll 4
    for (int t = 0; t < time_steps; t++) {
        const int current_idx = t * batch_stride + idx;
        const int next_idx = (t + 1) * batch_stride + idx;
        
        v = __hadd(v, x_seq[current_idx]);
        H_seq_out[next_idx] = v;
        
        __half spike = __float2half(0.0f);
        const bool pos_spike = __hge(v, local_v_th) & __hlt(T, local_T_max);
        const bool neg_spike = __hlt(v, __float2half(0.0f)) & __hgt(T, local_T_min);
        
        spike = pos_spike ? __float2half(1.0f) : (neg_spike ? __float2half(-1.0f) : __float2half(0.0f));
        
        const __half v_reset = __hmul(local_v_th, spike);
        const __half prefire_reset = __hlt(t, local_T_max) ? 
            __hdiv(__hmul(local_prefire, local_v_th), local_T_max) : __float2half(0.0f);
        
        v = __hsub(v, __hadd(v_reset, prefire_reset));
        T = __hadd(T, spike);
        
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    v_out[idx] = v;
}

// ========== Optimized backward kernels ==========
__global__ void __launch_bounds__(128, 8) backward_kernel_fp32(
    const float* __restrict__ grad_spike_seq,
    const float* __restrict__ grad_v,
    const float* __restrict__ grad_T_seq,
    const float* __restrict__ spike_seq,
    const float* __restrict__ T_seq,
    const float* __restrict__ H_seq,
    const float* v_th,
    const float* T_max,
    const float* T_min,
    float* __restrict__ grad_x_seq,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ float shared_backward[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_backward[0] = v_th[0];
        shared_backward[1] = T_max[0];
        shared_backward[2] = T_min[0];
    }
    __syncthreads();
    
    const float local_v_th = shared_backward[0];
    const float local_T_max = shared_backward[1];
    const float local_T_min = shared_backward[2];
    
    float grad_V = 0.0f;
    float grad_T = 0.0f;
    
    const int batch_stride = batch_size * features;
    
    for (int t = time_steps; t >= 1; t--) {
        const int current_idx = t * batch_stride + idx;
        const int prev_idx = (t-1) * batch_stride + idx;
        const int output_idx = current_idx - batch_stride;
        
        const float H_t = H_seq[current_idx];
        const float T_t_1 = T_seq[prev_idx];
        const float grad_Y_t = grad_spike_seq[output_idx];
        
        const float H_minus_vth = H_t - local_v_th;
        const float T_max_minus_T = local_T_max - T_t_1;
        const float T_minus_T_min = T_t_1 - local_T_min;
        
        const float grad_T_t_to_H_t = 
            theta_backward_fp32(H_minus_vth, local_v_th, T_t_1, local_T_min, local_T_max) * theta<float>(T_max_minus_T) +
            theta_backward_fp32(-H_t, local_v_th, T_t_1, local_T_min, local_T_max) * theta<float>(T_minus_T_min);

        const float grad_Y_t_to_T_t_1 = -(
            theta_eq<float>(H_minus_vth) * theta_backward_fp32(T_max_minus_T, 1.0f, T_t_1, local_T_min, local_T_max) +
            theta<float>(-H_t) * theta_backward_fp32(T_minus_T_min, 1.0f, T_t_1, local_T_min, local_T_max));
        
        const float common_term = grad_Y_t - local_v_th * grad_V + grad_T;
        const float grad_X = common_term * grad_T_t_to_H_t + grad_V;
        grad_T = common_term * grad_Y_t_to_T_t_1 + grad_T;
        grad_V = grad_X;
        
        grad_x_seq[output_idx] = grad_X;
    }
}

__global__ void __launch_bounds__(128, 8) backward_kernel_fp16(
    const __half* __restrict__ grad_spike_seq,
    const __half* __restrict__ grad_v,
    const __half* __restrict__ grad_T_seq,
    const __half* __restrict__ spike_seq,
    const __half* __restrict__ T_seq,
    const __half* __restrict__ H_seq,
    const __half* v_th,
    const __half* T_max,
    const __half* T_min,
    __half* __restrict__ grad_x_seq,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ __half shared_backward_half[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_backward_half[0] = v_th[0];
        shared_backward_half[1] = T_max[0];
        shared_backward_half[2] = T_min[0];
    }
    __syncthreads();
    
    const __half local_v_th = shared_backward_half[0];
    const __half local_T_max = shared_backward_half[1];
    const __half local_T_min = shared_backward_half[2];
    
    __half grad_V = __float2half(0.0f);
    __half grad_T = __float2half(0.0f);
    
    const int batch_stride = batch_size * features;
    
    for (int t = time_steps; t >= 1; t--) {
        const int current_idx = t * batch_stride + idx;
        const int prev_idx = (t-1) * batch_stride + idx;
        const int output_idx = current_idx - batch_stride;
        
        const __half H_t = H_seq[current_idx];
        const __half T_t_1 = T_seq[prev_idx];
        const __half grad_Y_t = grad_spike_seq[output_idx];
        
        const __half H_minus_vth = __hsub(H_t, local_v_th);
        const __half T_max_minus_T = __hsub(local_T_max, T_t_1);
        const __half T_minus_T_min = __hsub(T_t_1, local_T_min);
        
        const __half grad_T_t_to_H_t = __hadd(
            __hmul(theta_backward_fp16(H_minus_vth, local_v_th, T_t_1, local_T_min, local_T_max), 
                   theta<__half>(T_max_minus_T)),
            __hmul(theta_backward_fp16(__hneg(H_t), local_v_th, T_t_1, local_T_min, local_T_max), 
                   theta<__half>(T_minus_T_min)));

        const __half grad_Y_t_to_T_t_1 = __hneg(__hadd(
            __hmul(theta_eq<__half>(H_minus_vth), 
                   theta_backward_fp16(T_max_minus_T, __float2half(1.0f), T_t_1, local_T_min, local_T_max)),
            __hmul(theta<__half>(__hneg(H_t)), 
                   theta_backward_fp16(T_minus_T_min, __float2half(1.0f), T_t_1, local_T_min, local_T_max))));
        
        const __half common_term = __hadd(__hsub(grad_Y_t, __hmul(local_v_th, grad_V)), grad_T);
        const __half grad_X = __hadd(__hmul(common_term, grad_T_t_to_H_t), grad_V);
        grad_T = __hadd(__hmul(common_term, grad_Y_t_to_T_t_1), grad_T);
        grad_V = grad_X;
        
        grad_x_seq[output_idx] = grad_X;
    }
}

__global__ void __launch_bounds__(128, 8) forward_kernel_fp64(
    const double* __restrict__ x_seq,
    const double* v_th,
    const double* T_max,
    const double* T_min,
    const double* prefire,
    double* __restrict__ spike_seq_out,
    double* __restrict__ v_out,
    double* __restrict__ T_seq_out,
    double* __restrict__ H_seq_out,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ double shared_data_double[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_data_double[0] = v_th[0];
        shared_data_double[1] = T_max[0];
        shared_data_double[2] = T_min[0];
        shared_data_double[3] = prefire[0];
    }
    __syncthreads();
    
    const double local_v_th = shared_data_double[0];
    const double local_T_max = shared_data_double[1];
    const double local_T_min = shared_data_double[2];
    const double local_prefire = shared_data_double[3];
    
    double v = (0.5 + local_prefire) * local_v_th;
    double T = 0.0;
    
    T_seq_out[idx] = T;
    spike_seq_out[idx] = 0.0;
    H_seq_out[idx] = v;
    
    const int batch_stride = batch_size * features;
    
    #pragma unroll 4
    for (int t = 0; t < time_steps; t++) {
        const int current_idx = t * batch_stride + idx;
        const int next_idx = (t + 1) * batch_stride + idx;
        
        v += x_seq[current_idx];
        H_seq_out[next_idx] = v;
        
        double spike = 0.0;
        const bool pos_spike = (v >= local_v_th) & (T < local_T_max);
        const bool neg_spike = (v < 0.0) & (T > local_T_min);
        
        spike = pos_spike ? 1.0 : (neg_spike ? -1.0 : 0.0);
        
        const double v_reset = local_v_th * spike;
        const double prefire_reset = (t < local_T_max) ? 
            local_prefire * local_v_th / local_T_max : 0.0;
        
        v -= (v_reset + prefire_reset);
        T += spike;
        
        spike_seq_out[next_idx] = spike;
        T_seq_out[next_idx] = T;
    }
    
    v_out[idx] = v;
}

__global__ void __launch_bounds__(128, 8) backward_kernel_fp64(
    const double* __restrict__ grad_spike_seq,
    const double* __restrict__ grad_v,
    const double* __restrict__ grad_T_seq,
    const double* __restrict__ spike_seq,
    const double* __restrict__ T_seq,
    const double* __restrict__ H_seq,
    const double* v_th,
    const double* T_max,
    const double* T_min,
    double* __restrict__ grad_x_seq,
    int batch_size,
    int time_steps,
    int features
) {
    extern __shared__ double shared_backward_double[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    if (threadIdx.x == 0) {
        shared_backward_double[0] = v_th[0];
        shared_backward_double[1] = T_max[0];
        shared_backward_double[2] = T_min[0];
    }
    __syncthreads();
    
    const double local_v_th = shared_backward_double[0];
    const double local_T_max = shared_backward_double[1];
    const double local_T_min = shared_backward_double[2];
    
    double grad_V = 0.0;
    double grad_T = 0.0;
    
    const int batch_stride = batch_size * features;
    
    for (int t = time_steps; t >= 1; t--) {
        const int current_idx = t * batch_stride + idx;
        const int prev_idx = (t-1) * batch_stride + idx;
        const int output_idx = current_idx - batch_stride;
        
        const double H_t = H_seq[current_idx];
        const double T_t_1 = T_seq[prev_idx];
        const double grad_Y_t = grad_spike_seq[output_idx];
        
        const double H_minus_vth = H_t - local_v_th;
        const double T_max_minus_T = local_T_max - T_t_1;
        const double T_minus_T_min = T_t_1 - local_T_min;
        
        const double grad_T_t_to_H_t = 
            theta_backward_fp64(H_minus_vth, local_v_th, T_t_1, local_T_min, local_T_max) * theta<double>(T_max_minus_T) +
            theta_backward_fp64(-H_t, local_v_th, T_t_1, local_T_min, local_T_max) * theta<double>(T_minus_T_min);

        const double grad_Y_t_to_T_t_1 = -(
            theta_eq<double>(H_minus_vth) * theta_backward_fp64(T_max_minus_T, 1.0, T_t_1, local_T_min, local_T_max) +
            theta<double>(-H_t) * theta_backward_fp64(T_minus_T_min, 1.0, T_t_1, local_T_min, local_T_max));
        
        const double common_term = grad_Y_t - local_v_th * grad_V + grad_T;
        const double grad_X = common_term * grad_T_t_to_H_t + grad_V;
        grad_T = common_term * grad_Y_t_to_T_t_1 + grad_T;
        grad_V = grad_X;
        
        grad_x_seq[output_idx] = grad_X;
    }
}

} // extern "C"