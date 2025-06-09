#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

template <typename T>
__device__ __forceinline__
T zero() { return T(0); }

template <typename T>
__device__ __forceinline__
T one() { return T(1); }

template <typename T>
__device__ __forceinline__
T fast_exp(T x);

template <>
__device__ __forceinline__
float fast_exp(float x) { return __expf(x); }

template <>
__device__ __forceinline__
double fast_exp(double x) { return exp(x); }

template <>
__device__ __forceinline__
__half fast_exp(__half x) { return hexp(x); }

template <typename T>
__device__ __forceinline__
T theta(T x) { return x > zero<T>() ? one<T>() : zero<T>(); }

template <typename T>
__device__ __forceinline__
T theta_eq(T x) { return x >= zero<T>() ? one<T>() : zero<T>(); }

template <typename T>
__device__ __forceinline__
T theta_backward(T x, T Vthr)
{
    const T sigma2 = static_cast<T>(0.405) * Vthr;
    const T a = one<T>() / Vthr;
    const T upper = -(x * x) / (static_cast<T>(2.0) * sigma2 * sigma2);
    return a * fast_exp(upper);
}

template <typename T>
__global__ void forward_kernel(
    const T *__restrict__ x_seq,
    const T *__restrict__ v_th,
    const T *__restrict__ T_max,
    const T *__restrict__ T_min,
    const T *__restrict__ prefire,
    /* out */ T *__restrict__ spike_seq,
    T *__restrict__ v_out,
    T *__restrict__ T_seq,
    T *__restrict__ H_seq,
    /* scalars */ int N, int Tlen, int feat_flat, int total_neuron)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = gid; idx < total_neuron; idx += blockDim.x * gridDim.x)
    {
        const T Vthr = v_th[0];
        const T Tmax = T_max[0];
        const T Tmin = T_min[0];
        const T pre  = prefire[0];

        T v = (static_cast<T>(0.5) + pre) * Vthr;
        T Tcnt = zero<T>();

        T_seq[idx] = Tcnt;
        spike_seq[idx] = zero<T>();
        H_seq[idx] = v;

        const int stride = total_neuron;
        for (int t = 0; t < Tlen; ++t)
        {
            const int off    = t * stride + idx;
            const int w_off  = (t + 1) * stride + idx;

            v += __ldg(x_seq + off);
            H_seq[w_off] = v;

            const bool pos_spk = (v >= Vthr) & (Tcnt < Tmax);
            const bool neg_spk = (v < zero<T>()) & (Tcnt > Tmin);

            T spike = zero<T>();
            if (pos_spk)      spike = one<T>();
            else if (neg_spk) spike = -one<T>();

            T prefire_rst = zero<T>();
            if( t < static_cast<int>(Tmax))
                prefire_rst = pre * Vthr / Tmax;

            v -= Vthr * spike + prefire_rst;
            Tcnt += spike;

            spike_seq[w_off] = spike;
            T_seq[w_off]     = Tcnt;
        }
        v_out[idx] = v;
    }
}

template <typename T>
__global__ void backward_kernel(
    const T *__restrict__ grad_spike,
    const T *__restrict__ grad_v,
    const T *__restrict__ grad_Tseq,
    const T *__restrict__ spike_seq,
    const T *__restrict__ T_seq,
    const T *__restrict__ H_seq,
    const T *__restrict__ v_th,
    const T *__restrict__ T_max,
    const T *__restrict__ T_min,
    /* out */ T *__restrict__ grad_x,
    /* scalars */ int N, int Tlen, int feat_flat, int total_neuron)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = gid; idx < total_neuron; idx += blockDim.x * gridDim.x)
    {
        const T Vthr = v_th[0];
        const T Tmax = T_max[0];
        const T Tmin = T_min[0];

        const int stride = total_neuron;

        T gV = zero<T>();
        T gT = zero<T>();

        for (int t = Tlen; t >= 1; --t)
        {
            const int cur_off = t * stride + idx;
            const int pre_off = (t - 1) * stride + idx;

            const T H_t     = __ldg(H_seq + cur_off);
            const T T_prev  = __ldg(T_seq + pre_off);
            const T gY_t    = __ldg(grad_spike + pre_off);

            const T H_minus = H_t - Vthr;
            const T Tmax_mT = Tmax - T_prev;
            const T Tminus  = T_prev - Tmin;

            const T dYdH =   theta_backward(H_minus, Vthr) * theta(Tmax_mT)
                           + theta_backward(-H_t, Vthr)    * theta(Tminus);

            const T dYdTprev = - ( theta_eq(H_minus) * theta_backward(Tmax_mT, one<T>())
                                 + theta(-H_t)       * theta_backward(Tminus,  one<T>()) );

            const T common = gY_t - Vthr * gV + gT;

            T gX = common * dYdH + gV;
            gT   = common * dYdTprev + gT;
            gV   = gX;

            grad_x[pre_off] = gX;
        }
    }
}


extern "C" {

__global__ void forward_kernel_fp16(
    const __half* x, const __half* vth, const __half* Tmax, const __half* Tmin, const __half* pre,
    __half* spk, __half* vout, __half* Tseq, __half* Hseq,
    int N, int T, int F, int total)
{ forward_kernel(x, vth, Tmax, Tmin, pre, spk, vout, Tseq, Hseq, N, T, F, total); }

__global__ void backward_kernel_fp16(
    const __half* gspk, const __half* gv, const __half* gT,
    const __half* spk, const __half* Tseq, const __half* Hseq,
    const __half* vth, const __half* Tmax, const __half* Tmin,
    __half* gx, int N, int T, int F, int total)
{ backward_kernel(gspk, gv, gT, spk, Tseq, Hseq, vth, Tmax, Tmin, gx, N, T, F, total); }

__global__ void forward_kernel_fp32(
    const float* x, const float* vth, const float* Tmax, const float* Tmin, const float* pre,
    float* spk, float* vout, float* Tseq, float* Hseq,
    int N, int T, int F, int total)
{ forward_kernel(x, vth, Tmax, Tmin, pre, spk, vout, Tseq, Hseq, N, T, F, total); }

__global__ void backward_kernel_fp32(
    const float* gspk, const float* gv, const float* gT,
    const float* spk, const float* Tseq, const float* Hseq,
    const float* vth, const float* Tmax, const float* Tmin,
    float* gx, int N, int T, int F, int total)
{ backward_kernel(gspk, gv, gT, spk, Tseq, Hseq, vth, Tmax, Tmin, gx, N, T, F, total); }

__global__ void forward_kernel_fp64(
    const double* x, const double* vth, const double* Tmax, const double* Tmin, const double* pre,
    double* spk, double* vout, double* Tseq, double* Hseq,
    int N, int T, int F, int total)
{ forward_kernel(x, vth, Tmax, Tmin, pre, spk, vout, Tseq, Hseq, N, T, F, total); }

__global__ void backward_kernel_fp64(
    const double* gspk, const double* gv, const double* gT,
    const double* spk, const double* Tseq, const double* Hseq,
    const double* vth, const double* Tmax, const double* Tmin,
    double* gx, int N, int T, int F, int total)
{ backward_kernel(gspk, gv, gT, spk, Tseq, Hseq, vth, Tmax, Tmin, gx, N, T, F, total); }

} // extern "C"
} // anonymous namespace
