// Pipelined tiled FMHA (sm80) + CUTLASS GEMM+epilogue.
//
// fmha_fwd: Flash-style software pipeline
//   - cp.async double-buffer K/V
//   - WMMA 16x16x16 for QK^T and P@V (tensor cores)
//   - online softmax + causal/window/softcap/strided-bias in the QK epilogue
//   - 2D key-pad bias is [Z,Ks], not a left-unsqueezed 4D view (Longformer IMA)
//
// scaled_gemm: GemmBatched + LinearCombination  D = α A Bᵀ + β C
// addmm_epilogue: device::Gemm + LinearCombination{GELU,ReLU}
//   D = act(A @ B + bias)   bias broadcast via ldc=0

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector_types.h>
#include <cstdint>
#include <vector>

#include <cutlass/half.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/gemm/gemm.h>

using namespace nvcuda;

using GemmBias = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>;

using GemmGelu = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

using GemmRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationRelu<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

using GemmPlain = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

__device__ __forceinline__ void cp_async_16(void* smem, const void* glob) {
    unsigned smem_ptr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_ptr), "l"(glob));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait0() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
}

template <int ROWS, int D>
__device__ void load_tile_async(half* smem, const half* gmem, int row0, int valid) {
    constexpr int vecs = ROWS * D / 8;
#pragma unroll 4
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        int e = i * 8;
        int r = e / D;
        int c = e % D;
        half* dst = smem + e;
        int gr = row0 + r;
        if (gr < valid) {
            cp_async_16(dst, gmem + static_cast<int64_t>(gr) * D + c);
        } else {
            *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
        }
    }
}

__host__ __device__ __forceinline__ int align16(int n) {
    return (n + 15) & ~15;
}

static int smem_bytes(int D) {
    int h = (int)sizeof(half);
    int f = (int)sizeof(float);
    int n = 0;
    n = align16(n + 64 * D * h);
    n = align16(n + 2 * 64 * D * h);
    n = align16(n + 2 * 64 * D * h);
    n = align16(n + 64 * D * f);
    n = align16(n + 64 * 64 * f);
    n = align16(n + 64 * 64 * h);
    n = align16(n + 64 * 16 * f);
    return n;
}

template <int D>
__global__ void gko_fmha_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ bias,
    half* __restrict__ O,
    float* __restrict__ LSE,
    int H,
    int Qs,
    int Ks,
    float scale,
    int causal,
    int window,
    float softcap,
    int64_t b_sb,
    int64_t b_sh,
    int64_t b_sq,
    int64_t b_sk,
    int bias_mode) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    const int q0 = blockIdx.x * BM;
    const int bh = blockIdx.y;
    const int z = bh / H;
    const int h = bh % H;
    if (q0 >= Qs) {
        return;
    }

    extern __shared__ __align__(16) unsigned char raw[];
    int off = 0;
    half* q_smem = reinterpret_cast<half*>(raw + off);
    off = align16(off + BM * D * (int)sizeof(half));
    half* k_bufs[2];
    k_bufs[0] = reinterpret_cast<half*>(raw + off);
    off = align16(off + BN * D * (int)sizeof(half));
    k_bufs[1] = reinterpret_cast<half*>(raw + off);
    off = align16(off + BN * D * (int)sizeof(half));
    half* v_bufs[2];
    v_bufs[0] = reinterpret_cast<half*>(raw + off);
    off = align16(off + BN * D * (int)sizeof(half));
    v_bufs[1] = reinterpret_cast<half*>(raw + off);
    off = align16(off + BN * D * (int)sizeof(half));
    float* o_smem = reinterpret_cast<float*>(raw + off);
    off = align16(off + BM * D * (int)sizeof(float));
    float* s_smem = reinterpret_cast<float*>(raw + off);
    off = align16(off + BM * BN * (int)sizeof(float));
    half* p_smem = reinterpret_cast<half*>(raw + off);
    off = align16(off + BM * BN * (int)sizeof(half));
    float* pv_tile = reinterpret_cast<float*>(raw + off);

    const int64_t head = static_cast<int64_t>(bh);
    const half* q_base = Q + head * Qs * D;
    const half* k_base = K + head * Ks * D;
    const half* v_base = V + head * Ks * D;
    half* o_base = O + head * Qs * D;
    float* lse_base = LSE + head * Qs;

    load_tile_async<BM, D>(q_smem, q_base, q0, Qs);
    cp_async_commit();
    cp_async_wait0();
    __syncthreads();

    for (int i = threadIdx.x; i < BM * D; i += blockDim.x) {
        o_smem[i] = 0.f;
    }
    __shared__ float row_m[BM];
    __shared__ float row_l[BM];
    if (threadIdx.x < BM) {
        row_m[threadIdx.x] = -1e30f;
        row_l[threadIdx.x] = 0.f;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int wr = warp * 16;
    int stage = 0;
    load_tile_async<BN, D>(k_bufs[0], k_base, 0, Ks);
    load_tile_async<BN, D>(v_bufs[0], v_base, 0, Ks);
    cp_async_commit();
    cp_async_wait0();
    __syncthreads();

    for (int n0 = 0; n0 < Ks; n0 += BN) {
        int next = n0 + BN;
        if (next < Ks) {
            int nxt = stage ^ 1;
            load_tile_async<BN, D>(k_bufs[nxt], k_base, next, Ks);
            load_tile_async<BN, D>(v_bufs[nxt], v_base, next, Ks);
            cp_async_commit();
        }

        half* k_smem = k_bufs[stage];
        half* v_smem = v_bufs[stage];

        for (int nc = 0; nc < BN; nc += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fr;
            wmma::fill_fragment(c_fr, 0.f);
            for (int d0 = 0; d0 < D; d0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fr;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fr;
                wmma::load_matrix_sync(a_fr, q_smem + wr * D + d0, D);
                wmma::load_matrix_sync(b_fr, k_smem + nc * D + d0, D);
                wmma::mma_sync(c_fr, a_fr, b_fr, c_fr);
            }
            wmma::store_matrix_sync(s_smem + wr * BN + nc, c_fr, BN, wmma::mem_row_major);
        }
        __syncwarp();

        const int lane = threadIdx.x & 31;
        const int row = wr + (lane % 16);
        const int col_hi = (lane / 16) * 32;
        const int qr = q0 + row;
        float tmax = -1e30f;
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            int c = col_hi + t;
            int kc = n0 + c;
            float s = s_smem[row * BN + c] * scale;
            bool keep = (qr < Qs) && (kc < Ks);
            if (causal && kc > qr) {
                keep = false;
            }
            if (window > 0) {
                int dist = kc - qr;
                if (dist > window || dist < -window) {
                    keep = false;
                }
            }
            if (bias_mode && keep) {
                int64_t bo;
                if (bias_mode == 1) {
                    bo = static_cast<int64_t>(z) * b_sb + static_cast<int64_t>(kc) * b_sk;
                } else {
                    bo = static_cast<int64_t>(z) * b_sb + static_cast<int64_t>(h) * b_sh +
                        static_cast<int64_t>(qr) * b_sq + static_cast<int64_t>(kc) * b_sk;
                }
                s += __half2float(bias[bo]);
            }
            if (softcap > 0.f && keep) {
                float x = s / softcap;
                x = fminf(fmaxf(x, -15.f), 15.f);
                s = softcap * tanhf(x);
            }
            if (!keep) {
                s = -1e30f;
            }
            s_smem[row * BN + c] = s;
            tmax = fmaxf(tmax, s);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, 16));
        float m_new = fmaxf(row_m[row], tmax);
        float alpha = (row_m[row] > -1e29f) ? expf(row_m[row] - m_new) : 0.f;
        if (col_hi == 0) {
#pragma unroll 8
            for (int d = 0; d < D; ++d) {
                o_smem[row * D + d] *= alpha;
            }
        }
        __syncwarp();
        float tsum = 0.f;
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            int c = col_hi + t;
            float p = expf(s_smem[row * BN + c] - m_new);
            s_smem[row * BN + c] = p;
            tsum += p;
        }
        tsum += __shfl_xor_sync(0xffffffff, tsum, 16);
        if (col_hi == 0) {
            row_l[row] = row_l[row] * alpha + tsum;
            row_m[row] = m_new;
        }
        __syncwarp();

        for (int i = lane; i < 16 * BN; i += 32) {
            p_smem[wr * BN + i] = __float2half(s_smem[wr * BN + i]);
        }
        __syncwarp();

        for (int d0 = 0; d0 < D; d0 += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fr;
            wmma::fill_fragment(c_fr, 0.f);
            for (int k0 = 0; k0 < BN; k0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fr;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_fr;
                wmma::load_matrix_sync(a_fr, p_smem + wr * BN + k0, BN);
                wmma::load_matrix_sync(b_fr, v_smem + k0 * D + d0, D);
                wmma::mma_sync(c_fr, a_fr, b_fr, c_fr);
            }
            wmma::store_matrix_sync(pv_tile + wr * 16, c_fr, 16, wmma::mem_row_major);
            for (int i = lane; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                o_smem[(wr + r) * D + d0 + c] += pv_tile[wr * 16 + i];
            }
            __syncwarp();
        }
        if (next < Ks) {
            cp_async_wait0();
            __syncthreads();
        }
        stage ^= 1;
    }

    for (int r = threadIdx.x; r < BM; r += blockDim.x) {
        int qr = q0 + r;
        if (qr >= Qs) {
            continue;
        }
        float inv = (row_l[r] > 0.f) ? (1.f / row_l[r]) : 0.f;
        lse_base[qr] = row_m[r] + logf(fmaxf(row_l[r], 1e-20f));
        for (int d = 0; d < D; ++d) {
            o_base[qr * D + d] = __float2half(o_smem[r * D + d] * inv);
        }
    }
}

std::vector<at::Tensor> cutlass_fmha_fwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor bias,
    double scale,
    int64_t causal,
    int64_t window,
    double softcap) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "q fp16 cuda [Z,H,S,D]");
    TORCH_CHECK(q.dim() == 4);
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    const int Z = (int)q.size(0);
    const int H = (int)q.size(1);
    const int Qs = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Ks = (int)k.size(2);
    auto out = at::empty_like(q);
    auto lse = at::empty({Z * (int64_t)H, Qs}, q.options().dtype(at::kFloat));

    int bias_mode = 0;
    int64_t b_sb = 0, b_sh = 0, b_sq = 0, b_sk = 0;
    const half* bp = nullptr;
    at::Tensor bias_keep;
    if (bias.defined() && bias.numel() > 0) {
        auto b = bias.to(q.dtype());
        if ((b.dim() <= 2 && b.numel() == (int64_t)Z * Ks) ||
            (b.dim() == 3 && b.size(-1) == Ks && b.numel() == (int64_t)Z * Ks) ||
            (b.dim() == 4 && b.size(-2) == 1 && b.size(-1) == Ks)) {
            if (b.dim() == 4) {
                b = b.reshape({b.size(0) == Z ? Z : 1, Ks});
                if (b.size(0) == 1 && Z > 1) {
                    b = b.expand({Z, Ks});
                }
            }
            b = b.reshape({Z, Ks}).contiguous();
            bias_mode = 1;
            b_sb = b.stride(0);
            b_sk = b.stride(1);
            bias_keep = b;
            bp = reinterpret_cast<const half*>(bias_keep.data_ptr<at::Half>());
        } else {
            while (b.dim() < 4) {
                if (b.dim() == 2) {
                    b = b.view({b.size(0), 1, 1, b.size(1)});
                } else {
                    b = b.unsqueeze(1);
                }
            }
            bias_mode = 2;
            if (b.size(0) == 1 && Z > 1) {
                b_sb = 0;
            } else {
                b_sb = b.stride(0);
            }
            b_sh = (b.size(1) == 1) ? 0 : b.stride(1);
            b_sq = (b.size(2) == 1) ? 0 : b.stride(2);
            b_sk = b.stride(3);
            bias_keep = b;
            bp = reinterpret_cast<const half*>(bias_keep.data_ptr<at::Half>());
        }
    }

    dim3 grid((Qs + 63) / 64, Z * H);
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    auto kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    auto vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    auto op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    auto lp = lse.data_ptr<float>();

    auto launch = [&](auto kernel, int d) {
        int sm = smem_bytes(d);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        kernel<<<grid, block, sm, stream>>>(
            qp, kp, vp, bp, op, lp, H, Qs, Ks, (float)scale, (int)causal, (int)window,
            (float)softcap, b_sb, b_sh, b_sq, b_sk, bias_mode);
    };
    if (D == 64) {
        launch(gko_fmha_fwd_kernel<64>, 64);
    } else if (D == 128) {
        launch(gko_fmha_fwd_kernel<128>, 128);
    } else {
        TORCH_CHECK(false, "fmha_fwd supports D=64 or 128, got ", D);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

at::Tensor cutlass_scaled_gemm(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    double alpha,
    double beta) {
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "a[B,M,K] b[B,N,K]");
    a = a.contiguous();
    b = b.contiguous();
    const int batch = (int)a.size(0);
    const int M = (int)a.size(1);
    const int Kdim = (int)a.size(2);
    const int N = (int)b.size(1);
    TORCH_CHECK((int)b.size(2) == Kdim);
    auto d = at::empty({batch, M, N}, a.options());
    if (!(c.defined() && c.numel() > 0)) {
        c = d;
        beta = 0.0;
    } else {
        c = c.to(a.dtype()).contiguous();
        TORCH_CHECK(c.sizes() == d.sizes());
    }
    GemmBias gemm;
    typename GemmBias::Arguments args{
        {M, N, Kdim},
        {reinterpret_cast<cutlass::half_t*>(a.data_ptr<at::Half>()), Kdim},
        static_cast<int64_t>(M) * Kdim,
        {reinterpret_cast<cutlass::half_t*>(b.data_ptr<at::Half>()), Kdim},
        static_cast<int64_t>(N) * Kdim,
        {reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), N},
        static_cast<int64_t>(M) * N,
        {reinterpret_cast<cutlass::half_t*>(d.data_ptr<at::Half>()), N},
        static_cast<int64_t>(M) * N,
        {(float)alpha, (float)beta},
        batch,
    };
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess);
    TORCH_CHECK(gemm.initialize(args) == cutlass::Status::kSuccess);
    TORCH_CHECK(gemm() == cutlass::Status::kSuccess);
    return d;
}

template <typename Gemm>
at::Tensor run_addmm(at::Tensor bias, at::Tensor a, at::Tensor b, float alpha, float beta) {
    const int M = (int)a.size(0);
    const int Kdim = (int)a.size(1);
    const int N = (int)b.size(1);
    auto d = at::empty({M, N}, a.options());
    int ldc = N;
    cutlass::half_t* cptr;
    at::Tensor ckeep;
    if (!(bias.defined() && bias.numel() > 0) || beta == 0.f) {
        cptr = reinterpret_cast<cutlass::half_t*>(d.data_ptr<at::Half>());
        beta = 0.f;
        ldc = N;
    } else {
        auto bv = bias.reshape({-1}).contiguous();
        TORCH_CHECK(bv.numel() == N, "bias must be [N]");
        // Row-major C[m,n] = ptr[m*ldc + n]; ldc=0 broadcasts bias[n] to every row.
        cptr = reinterpret_cast<cutlass::half_t*>(bv.data_ptr<at::Half>());
        ldc = 0;
        ckeep = bv;
    }
    Gemm gemm;
    typename Gemm::Arguments args{
        {M, N, Kdim},
        {reinterpret_cast<cutlass::half_t*>(a.data_ptr<at::Half>()), Kdim},
        {reinterpret_cast<cutlass::half_t*>(b.data_ptr<at::Half>()), N},
        {cptr, ldc},
        {reinterpret_cast<cutlass::half_t*>(d.data_ptr<at::Half>()), N},
        {alpha, beta},
    };
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess && ldc == 0) {
        // Some pins reject ldc < N. Materialize a strided expand (no extra copy if
        // the kernel still runs); otherwise copy a full [M,N] bias.
        auto full = bias.reshape({1, N}).expand({M, N}).contiguous();
        ckeep = full;
        cptr = reinterpret_cast<cutlass::half_t*>(ckeep.data_ptr<at::Half>());
        ldc = N;
        args.ref_C = {cptr, ldc};
        st = gemm.can_implement(args);
    }
    TORCH_CHECK(st == cutlass::Status::kSuccess, "CUTLASS addmm cannot implement");
    TORCH_CHECK(gemm.initialize(args) == cutlass::Status::kSuccess);
    TORCH_CHECK(gemm() == cutlass::Status::kSuccess);
    return d;
}

at::Tensor cutlass_addmm_epilogue(
    at::Tensor bias,
    at::Tensor mat1,
    at::Tensor mat2,
    int64_t act) {
    TORCH_CHECK(mat1.is_cuda() && mat1.scalar_type() == at::kHalf);
    TORCH_CHECK(mat2.dim() == 2);
    mat2 = mat2.contiguous();
    auto lead = mat1.sizes().vec();
    TORCH_CHECK(!lead.empty());
    int64_t K = lead.back();
    TORCH_CHECK(mat2.size(0) == K);
    int64_t N = mat2.size(1);
    TORCH_CHECK(N % 8 == 0 && K % 8 == 0, "CUTLASS tensor-op addmm needs K,N % 8 == 0");
    at::Tensor a2 = mat1;
    if (mat1.dim() != 2) {
        a2 = mat1.contiguous().reshape({-1, K});
    } else if (!mat1.is_contiguous()) {
        a2 = mat1.contiguous();
    }
    float beta = (bias.defined() && bias.numel() > 0) ? 1.f : 0.f;
    at::Tensor out;
    if (act == 1) {
        out = run_addmm<GemmGelu>(bias, a2, mat2, 1.f, beta);
    } else if (act == 2) {
        out = run_addmm<GemmRelu>(bias, a2, mat2, 1.f, beta);
    } else {
        out = run_addmm<GemmPlain>(bias, a2, mat2, 1.f, beta);
    }
    if (mat1.dim() != 2) {
        lead.back() = N;
        out = out.reshape(lead);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmha_fwd", &cutlass_fmha_fwd);
    m.def("scaled_gemm", &cutlass_scaled_gemm);
    m.def("addmm_epilogue", &cutlass_addmm_epilogue);
}
