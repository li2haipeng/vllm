#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "vec_dtypes.cuh"
#include "kernel_config.h"

namespace cg = cooperative_groups;

// ============================================================================
// Multi-slice BGMV kernels for vLLM compatibility
// True BGMV style: per-token indices mapping, with slice parallelism via blockIdx.z
// ============================================================================

// Multi-slice shrink kernel (BGMV style with per-token indices)
// Input: X [num_tokens, feat_in]
// Output: Y [num_slices, num_tokens, feat_out]
// Weights: w_ptr[slice_id] -> [num_loras, feat_out, feat_in]
// Indices: indices[token_idx] -> lora_id for that token
// Grid: (feat_out, num_tokens, num_slices)
// Each block computes one output element for one token, one slice
template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty, typename in_T, typename out_T,
          typename W_T>
__global__ void
bgmv_shrink_sliced_kernel(out_T *__restrict__ Y,
                          const in_T *__restrict__ X,
                          W_T **__restrict__ w_ptr,
                          const int64_t *__restrict__ indices,
                          int64_t num_tokens,
                          float scale) {
  // blockIdx.z = slice_id, blockIdx.y = token_idx, blockIdx.x = output feature
  int slice_id = blockIdx.z;
  size_t token_idx = blockIdx.y;
  size_t j = blockIdx.x;  // output feature index
  
  // Get lora_id for this token
  int64_t lora_id = indices[token_idx];
  if (lora_id < 0) {
    return;  // No LoRA for this token
  }
  
  // Get weight pointer for this slice and lora
  // Weight layout: [num_loras, feat_out, feat_in]
  const W_T *W = w_ptr[slice_id] + (lora_id * feat_out + j) * feat_in;
  const in_T *X_token = X + token_idx * feat_in;
  
  auto block = cg::this_thread_block();
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ W_T W_shared[num_pipeline_stages * tile_size];
  __shared__ in_T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  // pipeline load W/X and compute WX
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
  cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     X_token + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
  pipe.producer_commit();
  
  size_t copy_idx, compute_idx;
  float y = 0.f;
  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size> w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size; ++tile_idx) {
    copy_idx = tile_idx % num_pipeline_stages;
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         W + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X_token + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
    }
    pipe.producer_commit();

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] = sum;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) {
      y += y_warpwise[i];
    }
    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  y_warpwise[threadIdx.y] =
      ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
          ? sum
          : 0.f;
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y += y_warpwise[i];
  }
  block.sync();
  pipe.consumer_release();

  // Write output: Y[slice_id, token_idx, j]
  if (block.thread_rank() == 0) {
    Y[slice_id * num_tokens * feat_out + token_idx * feat_out + j] += static_cast<out_T>(y);
  }
}


// Multi-slice expand kernel (BGMV style with per-token indices)
// Input: X [num_slices, num_tokens, feat_in]
// Output: Y [num_tokens, total_feat_out] (2D concatenated, vLLM format)
// Weights: w_ptr[slice_id] -> [num_loras, feat_out, feat_in]
// Indices: indices[token_idx] -> lora_id for that token
// slice_start_loc[slice_id] = column offset in output for this slice
// Grid: (feat_out / (ty * tz), num_tokens, num_slices)
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz,
          typename in_T, typename out_T, typename W_T>
__global__ void
bgmv_expand_sliced_kernel(out_T *__restrict__ Y,
                          const in_T *__restrict__ X,
                          W_T **__restrict__ w_ptr,
                          const int64_t *__restrict__ indices,
                          const int64_t *__restrict__ slice_start_loc,
                          int64_t num_tokens,
                          int64_t total_feat_out,
                          int32_t current_feat_out,
                          float scale) {
  // blockIdx.z = slice_id, blockIdx.y = token_idx, blockIdx.x = output tile
  int slice_id = blockIdx.z;
  size_t token_idx = blockIdx.y;
  size_t tile_idx = blockIdx.x;
  
  // Get lora_id for this token
  int64_t lora_id = indices[token_idx];
  if (lora_id < 0) {
    return;  // No LoRA for this token
  }
  
  // Get weight pointer for this slice and lora
  // Weight layout: [num_loras, feat_out, feat_in]
  const W_T *W = w_ptr[slice_id] + lora_id * current_feat_out * feat_in;
  
  // Get column offset for this slice
  int64_t col_offset = slice_start_loc[slice_id];
  
  auto block = cg::this_thread_block();
  
  // Load X for this token from slice input
  vec_t<in_T, vec_size> x_vec;
  x_vec.load(X + slice_id * num_tokens * feat_in + token_idx * feat_in + threadIdx.x * vec_size);

  // Load W
  vec_t<W_T, vec_size> w_vec;
  w_vec.load(W + (tile_idx * tz * ty) * feat_in + block.thread_rank() * vec_size);

  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }

  cg::thread_block_tile g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += g.shfl_down(sum, offset);
  }
  sum = g.shfl(sum, 0);

  // Write output: Y[token_idx, col_offset + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y]
  if (threadIdx.x == 0) {
    int out_col = col_offset + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y;
    Y[token_idx * total_feat_out + out_col] += static_cast<out_T>(sum);
  }
}

// Dispatch function for multi-slice shrink (BGMV style)
template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void bgmv_shrink_sliced(out_T *__restrict__ Y,
                        const in_T *__restrict__ X,
                        W_T **__restrict__ w_ptr,
                        const int64_t *__restrict__ indices,
                        int64_t num_tokens,
                        int64_t num_slices,
                        float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr size_t vec_size = ShrinkKernelConfig::vec_size;
  constexpr int cfg_tx = ShrinkKernelConfig::tx;
  constexpr int cfg_ty = ShrinkKernelConfig::ty;

  if constexpr (feat_in % (vec_size * cfg_tx) == 0) {
    constexpr int tx = cfg_tx;
    constexpr int ty = cfg_ty;

    // Grid: (feat_out, num_tokens, num_slices)
    dim3 nblks(feat_out, num_tokens, num_slices);
    dim3 nthrs(tx, ty);

    bgmv_shrink_sliced_kernel<feat_in, feat_out, vec_size, vec_size * sizeof(in_T),
                              vec_size * sizeof(W_T), tx, ty, in_T, out_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, indices, num_tokens, scale);
  } else if constexpr (feat_in % (vec_size / 2 * cfg_tx) == 0) {
    constexpr int tx = cfg_tx;
    constexpr int ty = cfg_ty;
    constexpr size_t half_vec = vec_size / 2;

    dim3 nblks(feat_out, num_tokens, num_slices);
    dim3 nthrs(tx, ty);

    bgmv_shrink_sliced_kernel<feat_in, feat_out, half_vec, half_vec * sizeof(in_T),
                              half_vec * sizeof(W_T), tx, ty, in_T, out_T, W_T>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, indices, num_tokens, scale);
  }
}

// Dispatch function for multi-slice expand (BGMV style)
template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void bgmv_expand_sliced(out_T *__restrict__ Y,
                        const in_T *__restrict__ X,
                        W_T **__restrict__ w_ptr,
                        const int64_t *__restrict__ indices,
                        const int64_t *__restrict__ slice_start_loc,
                        int64_t num_tokens,
                        int64_t num_slices,
                        int64_t total_feat_out,
                        int32_t current_feat_out,
                        float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr size_t vec_size = ExpandKernelConfig::vec_size;
  constexpr int tz = ExpandKernelConfig::tz;
  
  static_assert(feat_in % vec_size == 0, "feat_in must be divisible by vec_size");
  constexpr int tx = feat_in / vec_size;

  if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
    constexpr int ty = 32 / tx;
    // Grid: (feat_out / (ty * tz), num_tokens, num_slices)
    dim3 nblks(feat_out / (ty * tz), num_tokens, num_slices);
    dim3 nthrs(tx, ty, tz);

    bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, indices, slice_start_loc,
                                      num_tokens, total_feat_out,
                                      current_feat_out, scale);
  } else if constexpr (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
    constexpr int ty = 16 / tx;
    dim3 nblks(feat_out / (ty * tz), num_tokens, num_slices);
    dim3 nthrs(tx, ty, tz);

    bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, indices, slice_start_loc,
                                      num_tokens, total_feat_out,
                                      current_feat_out, scale);
  } else if constexpr (8 % tx == 0 && feat_out % (8 / tx * tz) == 0) {
    constexpr int ty = 8 / tx;
    dim3 nblks(feat_out / (ty * tz), num_tokens, num_slices);
    dim3 nthrs(tx, ty, tz);

    bgmv_expand_sliced_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
        <<<nblks, nthrs, 0, stream>>>(Y, X, w_ptr, indices, slice_start_loc,
                                      num_tokens, total_feat_out,
                                      current_feat_out, scale);
  }
}

// Instantiation macros for multi-slice kernels
#define INST_BGMV_SHRINK_SLICED(feat_in, feat_out, in_T, out_T, W_T)           \
  template void bgmv_shrink_sliced<feat_in, feat_out>(                         \
      out_T * __restrict__ Y, const in_T *__restrict__ X,                      \
      W_T **__restrict__ w_ptr, const int64_t *__restrict__ indices,           \
      int64_t num_tokens, int64_t num_slices, float scale);

#define INST_BGMV_EXPAND_SLICED(feat_in, feat_out, in_T, out_T, W_T)           \
  template void bgmv_expand_sliced<feat_in, feat_out>(                         \
      out_T * __restrict__ Y, const in_T *__restrict__ X,                      \
      W_T **__restrict__ w_ptr, const int64_t *__restrict__ indices,           \
      const int64_t *__restrict__ slice_start_loc,                             \
      int64_t num_tokens, int64_t num_slices,                                  \
      int64_t total_feat_out, int32_t current_feat_out, float scale);


// ============================================================================
// Original BGMV kernels (kept for backward compatibility)
// ============================================================================

// Shrink kernel: reduces from feat_in to feat_out (feat_in > feat_out)
// nthrs = (tx, ty)
template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty, typename in_T, typename out_T,
          typename W_T>
__global__ void
bgmv_shrink_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                   const W_T *__restrict__ W,
                   const int64_t *__restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t num_layers, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;
  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t j = blockIdx.x;
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ W_T W_shared[num_pipeline_stages * tile_size];
  __shared__ in_T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  // pipeline load W/X and compute WX;
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W + (idx * feat_out + j) * feat_in +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
  cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     X + (batch_idx * feat_in) +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
  pipe.producer_commit();
  size_t copy_idx, compute_idx;
  float y = 0.f;
  vec_t<in_T, vec_size> x_vec;
  vec_t<W_T, vec_size> w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size;
       ++tile_idx) {
    copy_idx = tile_idx % num_pipeline_stages;
    // pipeline stage: async copy W fragment
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         W + (idx * feat_out + j) * feat_in +
                             tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X + (batch_idx * feat_in) + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
    }
    pipe.producer_commit();

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // pipeline stage: compute WX
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] = sum;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) {
      y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  // final pipeline stage
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  y_warpwise[threadIdx.y] =
      ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
          ? sum
          : 0.f;
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y += y_warpwise[i];
  }

  block.sync();
  pipe.consumer_release();

  // write Y;
  if (block.thread_rank() == 0) {
    Y[batch_idx * full_y_size + y_offset + j] += static_cast<out_T>(y);
  }
}

// Expand kernel: expands from feat_in to feat_out (feat_in < feat_out)
// nthrs = (tx, ty, tz)
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz,
          typename in_T, typename out_T, typename W_T>
__global__ void
bgmv_expand_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                   const W_T *__restrict__ W,
                   const int64_t *__restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t num_layers, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;

  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t tile_idx = blockIdx.x;

  // load X;
  vec_t<in_T, vec_size> x_vec;
  x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size);

  // load W;
  vec_t<W_T, vec_size> w_vec;
  w_vec.load(W + (idx * feat_out + tile_idx * tz * ty) * feat_in +
             block.thread_rank() * vec_size);

  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }

  cg::thread_block_tile g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += g.shfl_down(sum, offset);
  }
  sum = g.shfl(sum, 0);

  if (threadIdx.x == 0) {
    Y[batch_idx * full_y_size + y_offset + tile_idx * (tz * ty) +
      threadIdx.z * ty + threadIdx.y] += static_cast<out_T>(sum);
  }
}

// Dispatch function using kernel_config.h parameters
template <int feat_in, int feat_out, typename in_T, typename out_T,
          typename W_T>
void bgmv_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                 const W_T *__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size, int64_t num_layers,
                 int64_t layer_idx, float scale) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if constexpr (feat_in <= feat_out) {
    // Expand kernel - use ExpandKernelConfig
    constexpr size_t vec_size = ExpandKernelConfig::vec_size;
    constexpr int tz = ExpandKernelConfig::tz;
    
    static_assert(feat_in % vec_size == 0, "feat_in must be divisible by vec_size");
    constexpr int tx = feat_in / vec_size;

    static_assert((32 % tx == 0 && feat_out % (32 / tx * tz) == 0) ||
                  (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) ||
                  (8 % tx == 0 && feat_out % (8 / tx * tz) == 0),
                  "Invalid expand kernel configuration");

    if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
      constexpr int ty = 32 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else if constexpr (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
      constexpr int ty = 16 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else {
      constexpr int ty = 8 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    }
  } else {
    // Shrink kernel - use ShrinkKernelConfig
    constexpr size_t vec_size = ShrinkKernelConfig::vec_size;
    constexpr int cfg_tx = ShrinkKernelConfig::tx;
    constexpr int cfg_ty = ShrinkKernelConfig::ty;

    static_assert(feat_in % (vec_size * cfg_tx) == 0 ||
                  feat_in % (vec_size / 2 * cfg_tx) == 0,
                  "Invalid shrink kernel configuration: feat_in must be divisible by vec_size*tx or (vec_size/2)*tx");

    if constexpr (feat_in % (vec_size * cfg_tx) == 0) {
      // Full vector size path
      constexpr int tx = cfg_tx;
      constexpr int ty = cfg_ty;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size, vec_size * sizeof(in_T),
                         vec_size * sizeof(W_T), tx, ty, in_T, out_T, W_T>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    } else if constexpr (feat_in % (vec_size / 2 * cfg_tx) == 0) {
      // Half vector size path for smaller feat_in alignments
      constexpr int tx = cfg_tx;
      constexpr int ty = cfg_ty;
      constexpr size_t half_vec = vec_size / 2;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, half_vec, half_vec * sizeof(in_T),
                         half_vec * sizeof(W_T), tx, ty, in_T, out_T, W_T>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, num_layers, layer_idx,
                                        scale);
    }
  }
}

// ============================================================================
// Instantiation macros
// ============================================================================

// Original BGMV instantiation
#define INST_BGMV(feat_in, feat_out, in_T, out_T, W_T)                         \
  template void bgmv_kernel<feat_in, feat_out>(                                \
      out_T * __restrict__ Y, const in_T *__restrict__ X,                      \
      const W_T *__restrict__ W, const int64_t *__restrict__ indicies,         \
      int64_t y_offset, int64_t full_y_size, int64_t batch_size,               \
      int64_t num_layers, int64_t layer_idx, float scale);

#define INST_BGMV_ONESIDE(in_T, out_T, W_T, feat_in, feat_out)                 \
  INST_BGMV(feat_in, feat_out, in_T, out_T, W_T)

#define INST_BGMV_TWOSIDE(in_T, out_T, W_T, narrow, wide)                      \
  INST_BGMV(narrow, wide, in_T, out_T, W_T)                                    \
  INST_BGMV(wide, narrow, in_T, out_T, W_T)
