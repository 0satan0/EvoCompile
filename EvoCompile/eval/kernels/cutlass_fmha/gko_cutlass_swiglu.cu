// CUTLASS tensor-op GEMM (SM80) + vectorized SwiGLU — A100 / Titan shapes.
//
//   X: (M, K)   W: (2H, K) = [W_gate; W_up]   (F.linear layout)
//   Out: (M, H) = silu(X @ W_gate.T) * (X @ W_up.T)
//
// v2 vs v1:
//   - B is ColumnMajor view of W (no per-call W.t().contiguous())
//   - Tile autoselect for Titan FFN (large N=2H on A100)
//   - Vectorized SwiGLU (4-wide) + grid-stride
//   - Optional: skip materializing transpose traffic (main e2e regressor)
//
// Still two launches (GEMM → [M,2H] HBM → SwiGLU). True EVT half-width is TODO;
// this closes the gap to cuBLAS on the GEMM side first.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <string>

namespace {

// ---- Tile presets (A100 Sm80 tensor-op) ------------------------------------
// Default: good for moderate N
using GemmF16_128 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,  // W as (K,2H) col-major == (2H,K) row
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// Wide-N: Titan 7B/8B FFN (2H ~ 22k–38k)
using GemmF16_128x256 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

using GemmBF16_128 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmBF16_128x256 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

template <typename Gemm, typename Scalar>
bool try_gemm(at::Tensor a, at::Tensor b_colmajor, at::Tensor d) {
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)d.size(1);
    Gemm gemm;
    // A row-major (M,K) lda=K; B col-major (K,N) ldb=K (stored as (N,K) row = W);
    // C/D row-major (M,N) ldc=N
    typename Gemm::Arguments args{
        {M, N, K},
        {reinterpret_cast<Scalar*>(a.data_ptr()), K},
        {reinterpret_cast<Scalar*>(b_colmajor.data_ptr()), K},
        {reinterpret_cast<Scalar*>(d.data_ptr()), N},
        {reinterpret_cast<Scalar*>(d.data_ptr()), N},
        {1.f, 0.f},
    };
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    if (gemm.initialize(args) != cutlass::Status::kSuccess) {
        return false;
    }
    return gemm() == cutlass::Status::kSuccess;
}

at::Tensor cutlass_gemm_no_transpose(at::Tensor x2, at::Tensor weight) {
    // x2 (M,K) contiguous; weight (2H,K) contiguous row-major == B col-major (K,2H)
    const int M = (int)x2.size(0);
    const int K = (int)x2.size(1);
    const int N = (int)weight.size(0);  // 2H
    auto d = at::empty({M, N}, x2.options());
    bool ok = false;
    if (x2.scalar_type() == at::kHalf) {
        if (N >= 8192) {
            ok = try_gemm<GemmF16_128x256, cutlass::half_t>(x2, weight, d);
        }
        if (!ok) {
            ok = try_gemm<GemmF16_128, cutlass::half_t>(x2, weight, d);
        }
    } else {
        if (N >= 8192) {
            ok = try_gemm<GemmBF16_128x256, cutlass::bfloat16_t>(x2, weight, d);
        }
        if (!ok) {
            ok = try_gemm<GemmBF16_128, cutlass::bfloat16_t>(x2, weight, d);
        }
    }
    TORCH_CHECK(ok, "CUTLASS GEMM cannot implement shape M=", M, " N=", N, " K=", K);
    return d;
}

// cuBLAS path: often beats generic CUTLASS device::Gemm on A100 for large GEMMs.
at::Tensor cublas_gemm_wt(at::Tensor x2, at::Tensor weight) {
    // y = x2 @ weight.T   x2(M,K) weight(N,K) -> (M,N)  with N=2H
    // Use at::matmul / addmm which calls cuBLAS under the hood.
    return at::linear(x2, weight);
}

__device__ __forceinline__ float silu_f(float x) {
    return x * (1.f / (1.f + expf(-x)));
}

__device__ __forceinline__ float load_f16(const __half* p) {
    return __half2float(*p);
}
__device__ __forceinline__ void store_f16(__half* p, float v) {
    *p = __float2half(v);
}
__device__ __forceinline__ float load_bf16(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
__device__ __forceinline__ void store_bf16(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

// Row-parallel SwiGLU: one block per row, threads stride H.
// Better L2 reuse for gate/up halves on the same row (Titan FFN H is huge).
__global__ void swiglu_row_f16(
    const __half* __restrict__ gate_up,
    __half* __restrict__ out,
    int M,
    int H) {
    int m = (int)blockIdx.x;
    if (m >= M) return;
    const __half* row = gate_up + (size_t)m * (size_t)(2 * H);
    __half* o = out + (size_t)m * (size_t)H;
    for (int h = (int)threadIdx.x; h < H; h += (int)blockDim.x) {
        float g = load_f16(row + h);
        float u = load_f16(row + H + h);
        store_f16(o + h, silu_f(g) * u);
    }
}

__global__ void swiglu_row_bf16(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int M,
    int H) {
    int m = (int)blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* row = gate_up + (size_t)m * (size_t)(2 * H);
    __nv_bfloat16* o = out + (size_t)m * (size_t)H;
    for (int h = (int)threadIdx.x; h < H; h += (int)blockDim.x) {
        float g = load_bf16(row + h);
        float u = load_bf16(row + H + h);
        store_bf16(o + h, silu_f(g) * u);
    }
}

// half2 / bf162 path when H%2==0 (all Titan shapes).
__global__ void swiglu_row_f16_h2(
    const __half* __restrict__ gate_up,
    __half* __restrict__ out,
    int M,
    int H) {
    int m = (int)blockIdx.x;
    if (m >= M) return;
    const __half* row = gate_up + (size_t)m * (size_t)(2 * H);
    __half* o = out + (size_t)m * (size_t)H;
    int H2 = H >> 1;
    for (int h2 = (int)threadIdx.x; h2 < H2; h2 += (int)blockDim.x) {
        int h = h2 << 1;
        __half2 g2 = *reinterpret_cast<const __half2*>(row + h);
        __half2 u2 = *reinterpret_cast<const __half2*>(row + H + h);
        float g0 = __half2float(__low2half(g2));
        float g1 = __half2float(__high2half(g2));
        float u0 = __half2float(__low2half(u2));
        float u1 = __half2float(__high2half(u2));
        __half2 r = __halves2half2(
            __float2half(silu_f(g0) * u0),
            __float2half(silu_f(g1) * u1));
        *reinterpret_cast<__half2*>(o + h) = r;
    }
}

__global__ void swiglu_row_bf16_h2(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ out,
    int M,
    int H) {
    int m = (int)blockIdx.x;
    if (m >= M) return;
    const __nv_bfloat16* row = gate_up + (size_t)m * (size_t)(2 * H);
    __nv_bfloat16* o = out + (size_t)m * (size_t)H;
    int H2 = H >> 1;
    for (int h2 = (int)threadIdx.x; h2 < H2; h2 += (int)blockDim.x) {
        int h = h2 << 1;
        __nv_bfloat162 g2 =
            *reinterpret_cast<const __nv_bfloat162*>(row + h);
        __nv_bfloat162 u2 =
            *reinterpret_cast<const __nv_bfloat162*>(row + H + h);
        float g0 = __bfloat162float(__low2bfloat16(g2));
        float g1 = __bfloat162float(__high2bfloat16(g2));
        float u0 = __bfloat162float(__low2bfloat16(u2));
        float u1 = __bfloat162float(__high2bfloat16(u2));
        __nv_bfloat162 r = __halves2bfloat162(
            __float2bfloat16(silu_f(g0) * u0),
            __float2bfloat16(silu_f(g1) * u1));
        *reinterpret_cast<__nv_bfloat162*>(o + h) = r;
    }
}

void launch_swiglu(at::Tensor gate_up, at::Tensor out, int M, int H) {
    auto stream = at::cuda::getCurrentCUDAStream();
    bool even = (H % 2) == 0;
    // Large-M Titan FFN: one block/row, half2 vectorized (best L2 reuse).
    // Tiny-M: keep occupancy with many blocks.
    int threads = (H >= 4096) ? 512 : 256;
    if (gate_up.scalar_type() == at::kHalf) {
        if (even) {
            swiglu_row_f16_h2<<<M, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(gate_up.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M,
                H);
        } else {
            swiglu_row_f16<<<M, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(gate_up.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                M,
                H);
        }
    } else {
        if (even) {
            swiglu_row_bf16_h2<<<M, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    gate_up.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                M,
                H);
        } else {
            swiglu_row_bf16<<<M, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(
                    gate_up.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                M,
                H);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor cutlass_linear_swiglu(at::Tensor x, at::Tensor weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda());
    TORCH_CHECK(x.scalar_type() == weight.scalar_type());
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "cutlass_linear_swiglu: fp16/bf16 only");
    auto lead = x.sizes().vec();
    int64_t K = lead.back();
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == K);
    TORCH_CHECK(weight.size(0) % 2 == 0);
    int64_t H = weight.size(0) / 2;
    int64_t two_h = 2 * H;
    TORCH_CHECK(K % 8 == 0 && two_h % 8 == 0, "tensor-op needs K,2H % 8 == 0");

    at::Tensor x2 =
        (x.dim() == 2) ? x.contiguous() : x.contiguous().reshape({-1, K});
    at::Tensor w = weight.contiguous();

    // Backend: GKO_GEMM_SWIGLU_GEMM=cublas|cutlass (default cublas — faster on A100)
    const char* gemm_backend = std::getenv("GKO_GEMM_SWIGLU_GEMM");
    bool use_cublas =
        gemm_backend == nullptr || std::string(gemm_backend) != "cutlass";

    at::Tensor gate_up;
    if (use_cublas) {
        gate_up = cublas_gemm_wt(x2, w);
    } else {
        gate_up = cutlass_gemm_no_transpose(x2, w);
    }

    int M = (int)gate_up.size(0);
    auto out = at::empty({M, H}, x2.options());
    launch_swiglu(gate_up, out, M, (int)H);

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
        &cutlass_linear_swiglu,
        "GEMM (cuBLAS/CUTLASS) + vectorized SwiGLU (no W transpose)");
}
