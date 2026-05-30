#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxHeadDim = 256;
constexpr int kTileSize = 16;

__device__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float warp_reduce_max(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float block_reduce_sum(float value) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ float block_reduce_max(float value) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -INFINITY;
    if (warp == 0) {
        value = warp_reduce_max(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

__global__ void standard_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
    __shared__ float q_shared[kMaxHeadDim];
    __shared__ float out_acc[kMaxHeadDim];
    __shared__ float running_max_shared;
    __shared__ float running_sum_shared;
    __shared__ float old_output_scale_shared;
    __shared__ float new_value_weight_shared;

    int bh = blockIdx.x;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int tid = threadIdx.x;

    const float* q_ptr = q + (batch * num_heads + head) * head_dim;
    const float* k_ptr = k + ((batch * num_heads + head) * seq_len) * head_dim;
    const float* v_ptr = v + ((batch * num_heads + head) * seq_len) * head_dim;
    float* out_ptr = out + (batch * num_heads + head) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q_ptr[d];
        out_acc[d] = 0.0f;
    }
    if (tid == 0) {
        running_max_shared = -INFINITY;
        running_sum_shared = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < seq_len; ++token) {
        float local_dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            local_dot += q_shared[d] * k_ptr[token * head_dim + d];
        }
        float score = block_reduce_sum(local_dot) * scale;

        if (tid == 0) {
            float old_max = running_max_shared;
            float old_sum = running_sum_shared;
            float new_max = fmaxf(old_max, score);
            float old_scale = (old_sum == 0.0f) ? 0.0f : expf(old_max - new_max);
            float new_weight = expf(score - new_max);

            running_max_shared = new_max;
            running_sum_shared = old_sum * old_scale + new_weight;
            old_output_scale_shared = old_scale;
            new_value_weight_shared = new_weight;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_acc[d] =
                out_acc[d] * old_output_scale_shared + new_value_weight_shared * v_ptr[token * head_dim + d];
        }
        __syncthreads();
    }

    float normalizer = running_sum_shared;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = out_acc[d] / normalizer;
    }
}

__global__ void flash_decode_splitkv_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ partial_out,
    float* __restrict__ tile_maxes,
    float* __restrict__ tile_sums,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int num_tiles,
    float scale) {
    extern __shared__ float shared[];
    float* q_shared = shared;
    float* k_tile = q_shared + head_dim;
    float* v_tile = k_tile + kTileSize * head_dim;
    float* scores = v_tile + kTileSize * head_dim;

    int bh = blockIdx.x;
    int tile = blockIdx.y;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int tid = threadIdx.x;
    int token_start = tile * kTileSize;
    int tile_tokens = min(kTileSize, seq_len - token_start);

    const float* q_ptr = q + (batch * num_heads + head) * head_dim;
    const float* k_ptr = k + ((batch * num_heads + head) * seq_len + token_start) * head_dim;
    const float* v_ptr = v + ((batch * num_heads + head) * seq_len + token_start) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q_ptr[d];
    }
    for (int idx = tid; idx < tile_tokens * head_dim; idx += blockDim.x) {
        k_tile[idx] = k_ptr[idx];
        v_tile[idx] = v_ptr[idx];
    }
    __syncthreads();

    for (int token = 0; token < tile_tokens; ++token) {
        float local_dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            local_dot += q_shared[d] * k_tile[token * head_dim + d];
        }
        float score = block_reduce_sum(local_dot) * scale;
        if (tid == 0) {
            scores[token] = score;
        }
        __syncthreads();
    }

    float local_max = (tid < tile_tokens) ? scores[tid] : -INFINITY;
    float tile_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    if (tid < tile_tokens) {
        float weight = expf(scores[tid] - tile_max);
        scores[tid] = weight;
        local_sum += weight;
    }
    float tile_sum = block_reduce_sum(local_sum);

    float* partial_ptr = partial_out + ((bh * num_tiles + tile) * head_dim);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int token = 0; token < tile_tokens; ++token) {
            acc += scores[token] * v_tile[token * head_dim + d];
        }
        partial_ptr[d] = acc / tile_sum;
    }

    if (tid == 0) {
        tile_maxes[bh * num_tiles + tile] = tile_max;
        tile_sums[bh * num_tiles + tile] = tile_sum;
    }
}

__global__ void flash_decode_combine_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ tile_maxes,
    const float* __restrict__ tile_sums,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_tiles) {
    int bh = blockIdx.x;
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int tile = tid; tile < num_tiles; tile += blockDim.x) {
        local_max = fmaxf(local_max, tile_maxes[bh * num_tiles + tile]);
    }
    float global_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int tile = tid; tile < num_tiles; tile += blockDim.x) {
        float tile_sum = tile_sums[bh * num_tiles + tile];
        float tile_max = tile_maxes[bh * num_tiles + tile];
        local_sum += tile_sum * expf(tile_max - global_max);
    }
    float global_sum = block_reduce_sum(local_sum);

    float* out_ptr = out + (batch * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int tile = 0; tile < num_tiles; ++tile) {
            float tile_weight = tile_sums[bh * num_tiles + tile] *
                                expf(tile_maxes[bh * num_tiles + tile] - global_max) / global_sum;
            acc += tile_weight * partial_out[(bh * num_tiles + tile) * head_dim + d];
        }
        out_ptr[d] = acc;
    }
}

void check_tensor_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32,
                "this beginner version supports float32 only");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4,
                "expected q [B,H,D], k [B,H,S,D], v [B,H,S,D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(2) == k.size(3),
                "shape mismatch: q [B,H,D], k/v [B,H,S,D]");
    TORCH_CHECK(q.size(2) <= kMaxHeadDim, "head_dim must be <= ", kMaxHeadDim);
}

torch::Tensor launch_standard_decode_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_tensor_inputs(q, k, v);

    auto out = torch::empty_like(q);
    int batch_size = static_cast<int>(q.size(0));
    int num_heads = static_cast<int>(q.size(1));
    int head_dim = static_cast<int>(q.size(2));
    int seq_len = static_cast<int>(k.size(2));
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    int blocks = batch_size * num_heads;
    auto stream = at::cuda::getCurrentCUDAStream();
    standard_decode_kernel<<<blocks, kBlockSize, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor launch_flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    check_tensor_inputs(q, k, v);

    auto out = torch::empty_like(q);
    int batch_size = static_cast<int>(q.size(0));
    int num_heads = static_cast<int>(q.size(1));
    int head_dim = static_cast<int>(q.size(2));
    int seq_len = static_cast<int>(k.size(2));
    int num_tiles = (seq_len + kTileSize - 1) / kTileSize;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    int blocks = batch_size * num_heads;
    auto partial_out = torch::empty({blocks, num_tiles, head_dim}, q.options());
    auto tile_maxes = torch::empty({blocks, num_tiles}, q.options());
    auto tile_sums = torch::empty({blocks, num_tiles}, q.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 split_grid(blocks, num_tiles);
    int shared_bytes = (head_dim + 2 * kTileSize * head_dim + kTileSize) * sizeof(float);
    flash_decode_splitkv_kernel<<<split_grid, kBlockSize, shared_bytes, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        tile_maxes.data_ptr<float>(),
        tile_sums.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        num_tiles,
        scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    flash_decode_combine_kernel<<<blocks, kBlockSize, 0, stream>>>(
        partial_out.data_ptr<float>(),
        tile_maxes.data_ptr<float>(),
        tile_sums.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_heads,
        head_dim,
        num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_decode_attention", &launch_standard_decode_attention, "Standard CUDA decode attention");
    m.def("flash_attention", &launch_flash_attention, "Tiled split-KV FlashAttention decode kernel");
}
