// True GEMM+SwiGLU epilogue: gate/up accumulate on-chip, store only [M,H].
// No 2H HBM materialization (unlike cuBLAS+silu / opaque silu_and_mul).
//
// Mainloop is register-tiled FP32 accumulate over fp16/bf16 inputs. For a
// CUTLASS tensor-op mainloop + custom EVT SwiGLU, see follow-up in
// gko_cutlass_attn.cu (device::Gemm only has GELU/ReLU stock epilogues today).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + expf(-x));
}

template <typename T>
__device__ __forceinline__ float ld(const T* p);

template <>
__device__ __forceinline__ float ld<__half>(const __half* p) {
    return __half2float(*p);
}
template <>
__device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ void st(T* p, float v);

template <>
__device__ __forceinline__ void st<__half>(__half* p, float v) {
    *p = __float2half(v);
}
template <>
__device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

template <typename T>
__global__ void linear_swiglu_epilogue_kernel(
    const T* __restrict__ X,
    const T* __restrict__ W,
    T* __restrict__ Out,
    int M,
    int H,
    int K) {
    const int m0 = blockIdx.y * BM;
    const int h0 = blockIdx.x * BN;
    if (m0 >= M || h0 >= H) {
        return;
    }

    __shared__ float As[BM][BK + 4];
    __shared__ float Bg[BK][BN + 4];
    __shared__ float Bu[BK][BN + 4];

    const int tid = threadIdx.x;
    const int tr = (tid / 16) * 4;
    const int tc = (tid % 16) * 4;

    float acc_g[4][4];
    float acc_u[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc_g[i][j] = 0.f;
            acc_u[i][j] = 0.f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int r = i / BK, c = i % BK;
            int gm = m0 + r, gk = k0 + c;
            As[r][c] =
                (gm < M && gk < K) ? ld(X + (int64_t)gm * K + gk) : 0.f;
        }
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            int gk = k0 + r, gh = h0 + c;
            float vg = 0.f, vu = 0.f;
            if (gk < K && gh < H) {
                vg = ld(W + (int64_t)gh * K + gk);
                vu = ld(W + (int64_t)(H + gh) * K + gk);
            }
            Bg[r][c] = vg;
            Bu[r][c] = vu;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = As[tr + i][kk];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc_g[i][j] += a * Bg[kk][tc + j];
                    acc_u[i][j] += a * Bu[kk][tc + j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gm = m0 + tr + i;
            int gh = h0 + tc + j;
            if (gm < M && gh < H) {
                st(Out + (int64_t)gm * H + gh, silu_f(acc_g[i][j]) * acc_u[i][j]);
            }
        }
    }
}

at::Tensor gko_linear_swiglu_cuda(at::Tensor x, at::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda());
    TORCH_CHECK(x.scalar_type() == weight.scalar_type());
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "linear_swiglu epilogue: fp16/bf16 only");
    auto lead = x.sizes().vec();
    int64_t K = lead.back();
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == K);
    TORCH_CHECK(weight.size(0) % 2 == 0);
    int64_t H = weight.size(0) / 2;
    at::Tensor x2 =
        (x.dim() == 2) ? x.contiguous() : x.contiguous().reshape({-1, K});
    weight = weight.contiguous();
    int M = (int)x2.size(0);
    auto out = at::empty({M, H}, x2.options());
    dim3 block(256);
    dim3 grid((int)((H + BN - 1) / BN), (int)((M + BM - 1) / BM));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x2.scalar_type() == at::kHalf) {
        linear_swiglu_epilogue_kernel<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x2.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M,
            (int)H,
            (int)K);
    } else {
        linear_swiglu_epilogue_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            M,
            (int)H,
            (int)K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (x.dim() != 2) {
        lead.back() = H;
        out = out.reshape(lead);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_swiglu",
        &gko_linear_swiglu_cuda,
        "True GEMM+SwiGLU epilogue (no 2H HBM)");
}
