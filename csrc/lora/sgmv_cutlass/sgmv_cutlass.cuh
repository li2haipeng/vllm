// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "config.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cute/tensor.hpp"

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <typename Element, typename KernelConfig>
struct GemmConfiguration {
  using TileShape = typename KernelConfig::TileShape;
  using ClusterShape = typename KernelConfig::ClusterShape;
  using KernelSchedule = typename KernelConfig::KernelSchedule;
  using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
  using ElementAccumulator = float;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<Element>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<Element>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<Element>::value;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    Element, LayoutC *, AlignmentC,
    Element, LayoutD *, AlignmentC,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<Element, ElementAccumulator>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    Element, LayoutA *, AlignmentA,
    Element, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
};

using ExpandConfig = GemmConfiguration<cutlass::half_t, ExpandKernelConfig>;
using ExpandConfigBf16 = GemmConfiguration<cutlass::bfloat16_t, ExpandKernelConfig>;
using ShrinkConfig = GemmConfiguration<cutlass::half_t, ShrinkKernelConfig>;
using ShrinkConfigBf16 = GemmConfiguration<cutlass::bfloat16_t, ShrinkKernelConfig>;

template <typename T>
inline T *alloc_from_buf(void **buf, int n) {
  auto *p = (T *)*buf;
  *buf = (void *)(p + n);
  return p;
}

size_t sgmv_tmp_size_sliced_vllm(int num_loras, int num_slices) {
  int total_problems = num_loras * num_slices;
  constexpr auto per_problem_sz = sizeof(void *) * 3 +
                                   sizeof(typename ExpandConfig::StrideA) * 4 +
                                   sizeof(typename ProblemShape::UnderlyingProblemShape);
  return per_problem_sz * total_problems + sizeof(int32_t) * num_slices * 2;
}

template <typename GemmConfig, typename cutlass_t>
__global__ void precompute_sgmv_shrink_args(
    typename ProblemShape::UnderlyingProblemShape *all_problems,
    cutlass_t **ptr_y, cutlass_t **ptr_x, cutlass_t **ptr_w,
    typename GemmConfig::StrideA *stride_x,
    typename GemmConfig::StrideB *stride_w,
    typename GemmConfig::StrideC *stride_c,
    typename GemmConfig::StrideD *stride_y,
    cutlass_t *y, int64_t y_slice_stride,
    cutlass_t *x_sorted,
    cutlass_t **w,
    int64_t w_lora_stride,
    int32_t *lora_token_start_loc,
    int32_t *active_lora_ids,
    int num_lora_indices,
    int num_slices,
    int d_in,
    int d_out,
    int max_loras) {
  int problem_idx = blockIdx.x;
  int slice_id = problem_idx / num_lora_indices;
  int lora_idx = problem_idx % num_lora_indices;

  int lora_id = active_lora_ids[lora_idx];
  bool valid_lora = (lora_id >= 0 && lora_id < max_loras);
  
  int m = valid_lora ? (lora_token_start_loc[lora_idx + 1] - lora_token_start_loc[lora_idx]) : 0;
  int k = d_in;
  int n = d_out;
  int start_pos = lora_token_start_loc[lora_idx];

  all_problems[problem_idx] = {m, n, k};
  ptr_w[problem_idx] = valid_lora ? (w[slice_id] + lora_id * w_lora_stride) : nullptr;
  ptr_x[problem_idx] = x_sorted + start_pos * d_in;
  ptr_y[problem_idx] = y + slice_id * y_slice_stride + start_pos * d_out;

  stride_x[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideA{}, cute::make_shape(m, k, 1));
  stride_w[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideB{}, cute::make_shape(n, k, 1));
  stride_c[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideC{}, cute::make_shape(m, n, 1));
  stride_y[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideD{}, cute::make_shape(m, n, 1));
}

template <typename GemmConfig, typename DType>
bool run_sgmv_shrink_kernel(DType *y, int64_t y_slice_stride,
                            DType *x_sorted,
                            DType **w,
                            int64_t w_lora_stride,
                            int32_t *lora_token_start_loc,
                            int32_t *active_lora_ids,
                            void *tmp_d,
                            int num_lora_indices,
                            int num_slices,
                            int num_tokens,
                            int d_in,
                            int d_out,
                            int max_loras,
                            cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  using Gemm = typename GemmConfig::Gemm;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  int total_problems = num_lora_indices * num_slices;
  if (total_problems == 0 || num_tokens == 0) {
    return true;
  }

  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto stride_X = alloc_from_buf<StrideA>(&tmp_d, total_problems);
  auto stride_W = alloc_from_buf<StrideB>(&tmp_d, total_problems);
  auto stride_C = alloc_from_buf<StrideC>(&tmp_d, total_problems);
  auto stride_Y = alloc_from_buf<StrideD>(&tmp_d, total_problems);
  auto all_problems = alloc_from_buf<UnderlyingProblemShape>(&tmp_d, total_problems);

  // Check if we're in cudagraph capture mode
  cudaStreamCaptureStatus capture_status;
  cudaStreamIsCapturing(stream, &capture_status);
  bool is_capturing = (capture_status == cudaStreamCaptureStatusActive);

  // Only sync outside of cudagraph capture
  if (!is_capturing) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      return false;
    }
  }

  precompute_sgmv_shrink_args<GemmConfig, cutlass_t><<<total_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, stride_X, stride_W, stride_C, stride_Y,
      (cutlass_t *)y, y_slice_stride,
      (cutlass_t *)x_sorted,
      (cutlass_t **)w,
      w_lora_stride,
      lora_token_start_loc, active_lora_ids,
      num_lora_indices, num_slices, d_in, d_out, max_loras);

  cudaError_t precompute_err = cudaGetLastError();
  if (precompute_err != cudaSuccess) {
    return false;
  }
  
  // Only sync outside of cudagraph capture
  if (!is_capturing) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      return false;
    }
  }

  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {total_problems, all_problems, nullptr},
    {(const cutlass_t**)ptr_X, stride_X, (const cutlass_t**)ptr_W, stride_W},
    {{1.0f, 0.0f}, (const cutlass_t**)ptr_Y, stride_C, ptr_Y, stride_Y},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) {
      return false;
    }
  }

  if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  if (workspace) cudaFreeAsync(workspace, stream);
  return true;
}

template <typename GemmConfig, typename cutlass_t>
__global__ void precompute_sgmv_expand_args(
    typename ProblemShape::UnderlyingProblemShape *all_problems,
    cutlass_t **ptr_y, cutlass_t **ptr_x, cutlass_t **ptr_w,
    typename GemmConfig::StrideA *stride_x,
    typename GemmConfig::StrideB *stride_w,
    typename GemmConfig::StrideC *stride_c,
    typename GemmConfig::StrideD *stride_y,
    cutlass_t *y_sorted,
    int64_t y_sorted_stride,
    int64_t *slice_start_loc,
    cutlass_t *x, int64_t x_slice_stride,
    cutlass_t **w,
    int64_t *w_lora_strides,
    int32_t *lora_token_start_loc,
    int32_t *active_lora_ids,
    int32_t *d_out_per_slice,
    int num_lora_indices,
    int num_slices,
    int d_in,
    int max_loras) {
  int problem_idx = blockIdx.x;
  int slice_id = problem_idx / num_lora_indices;
  int lora_idx = problem_idx % num_lora_indices;

  int lora_id = active_lora_ids[lora_idx];
  bool valid_lora = (lora_id >= 0 && lora_id < max_loras);
  
  int m = valid_lora ? (lora_token_start_loc[lora_idx + 1] - lora_token_start_loc[lora_idx]) : 0;
  int k = d_in;
  int n = d_out_per_slice[slice_id];
  int start_pos = lora_token_start_loc[lora_idx];
  int64_t col_offset = slice_start_loc[slice_id];

  all_problems[problem_idx] = {m, n, k};
  ptr_w[problem_idx] = valid_lora ? (w[slice_id] + lora_id * w_lora_strides[slice_id]) : nullptr;
  ptr_x[problem_idx] = x + slice_id * x_slice_stride + start_pos * d_in;
  ptr_y[problem_idx] = y_sorted + start_pos * y_sorted_stride + col_offset;

  stride_x[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideA{}, cute::make_shape(m, k, 1));
  stride_w[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideB{}, cute::make_shape(n, k, 1));
  stride_c[problem_idx] = typename GemmConfig::StrideC{y_sorted_stride, cute::Int<1>{}, cute::Int<0>{}};
  stride_y[problem_idx] = typename GemmConfig::StrideD{y_sorted_stride, cute::Int<1>{}, cute::Int<0>{}};
}


// Vectorized scatter-add kernel for expand operation
template <typename T>
__global__ void scatter_add_kernel_vectorized(T *y, const T *y_sorted, 
                                              const int32_t *indices,
                                              int num_tokens, int total_d_out,
                                              int64_t y_stride, int64_t y_sorted_stride,
                                              bool add_inputs) {
  int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  
  int dst_token = indices[token_idx];
  const T *src_row = y_sorted + token_idx * y_sorted_stride;
  T *dst_row = y + dst_token * y_stride;
  
  constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(T);
  int num_vectors = total_d_out / VECTOR_SIZE;
  
  const float4 *src_vec = reinterpret_cast<const float4*>(src_row);
  float4 *dst_vec = reinterpret_cast<float4*>(dst_row);
  
  for (int vec_idx = threadIdx.x; vec_idx < num_vectors; vec_idx += blockDim.x) {
    float4 val = src_vec[vec_idx];
    if (add_inputs) {
      float4 existing = dst_vec[vec_idx];
      T *val_arr = reinterpret_cast<T*>(&val);
      T *existing_arr = reinterpret_cast<T*>(&existing);
      #pragma unroll
      for (int i = 0; i < VECTOR_SIZE; i++) {
        val_arr[i] = val_arr[i] + existing_arr[i];
      }
    }
    dst_vec[vec_idx] = val;
  }
  
  int base_col = num_vectors * VECTOR_SIZE;
  for (int col = base_col + threadIdx.x; col < total_d_out; col += blockDim.x) {
    if (add_inputs) {
      dst_row[col] += src_row[col];
    } else {
      dst_row[col] = src_row[col];
    }
  }
}

// Scalar scatter-add kernel for small sizes or alignment issues
template <typename T>
__global__ void scatter_add_kernel_scalar(T *y, const T *y_sorted, 
                                          const int32_t *indices,
                                          int num_tokens, int total_d_out,
                                          int64_t y_stride, int64_t y_sorted_stride,
                                          bool add_inputs) {
  int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  
  int dst_token = indices[token_idx];
  const T *src_row = y_sorted + token_idx * y_sorted_stride;
  T *dst_row = y + dst_token * y_stride;
  
  for (int col = threadIdx.x; col < total_d_out; col += blockDim.x) {
    if (add_inputs) {
      dst_row[col] += src_row[col];
    } else {
      dst_row[col] = src_row[col];
    }
  }
}

template <typename T>
void launch_scatter_add(T *y, const T *y_sorted, 
                        const int32_t *indices,
                        int num_tokens, int total_d_out,
                        int64_t y_stride, int64_t y_sorted_stride,
                        bool add_inputs, cudaStream_t stream) {
  constexpr int VECTOR_SIZE = sizeof(float4) / sizeof(T);
  
  bool src_aligned = (reinterpret_cast<uintptr_t>(y_sorted) % sizeof(float4)) == 0;
  bool dst_aligned = (reinterpret_cast<uintptr_t>(y) % sizeof(float4)) == 0;
  bool stride_aligned = (y_stride % VECTOR_SIZE == 0) && (y_sorted_stride % VECTOR_SIZE == 0);
  bool use_vectorized = src_aligned && dst_aligned && stride_aligned && (total_d_out >= VECTOR_SIZE);
  
  int block_size = 256;
  int num_blocks = num_tokens;
  
  if (use_vectorized) {
    scatter_add_kernel_vectorized<<<num_blocks, block_size, 0, stream>>>(
        y, y_sorted, indices, num_tokens, total_d_out, 
        y_stride, y_sorted_stride, add_inputs);
  } else {
    scatter_add_kernel_scalar<<<num_blocks, block_size, 0, stream>>>(
        y, y_sorted, indices, num_tokens, total_d_out, 
        y_stride, y_sorted_stride, add_inputs);
  }
}

template <typename GemmConfig, typename DType>
bool run_sgmv_expand_kernel(DType *y,
                            int64_t y_row_stride,
                            int64_t *slice_start_loc,
                            DType *x, int64_t x_slice_stride,
                            DType **w,
                            int64_t *w_lora_strides,
                            int32_t *lora_token_start_loc,
                            int32_t *active_lora_ids,
                            int32_t *token_indices_sorted,
                            int32_t *d_out_per_slice,
                            void *tmp_d,
                            DType *y_sorted,
                            int64_t y_sorted_stride,
                            int num_lora_indices,
                            int num_slices,
                            int num_tokens,
                            int d_in,
                            int max_loras,
                            bool add_inputs,
                            cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  using Gemm = typename GemmConfig::Gemm;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  int total_problems = num_lora_indices * num_slices;
  int total_d_out = static_cast<int>(y_row_stride);

  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto stride_X = alloc_from_buf<StrideA>(&tmp_d, total_problems);
  auto stride_W = alloc_from_buf<StrideB>(&tmp_d, total_problems);
  auto stride_C = alloc_from_buf<StrideC>(&tmp_d, total_problems);
  auto stride_Y = alloc_from_buf<StrideD>(&tmp_d, total_problems);
  auto all_problems = alloc_from_buf<UnderlyingProblemShape>(&tmp_d, total_problems);

  precompute_sgmv_expand_args<GemmConfig, cutlass_t><<<total_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, stride_X, stride_W, stride_C, stride_Y,
      (cutlass_t *)y_sorted, y_sorted_stride, slice_start_loc,
      (cutlass_t *)x, x_slice_stride,
      (cutlass_t **)w,
      w_lora_strides,
      lora_token_start_loc, active_lora_ids, d_out_per_slice,
      num_lora_indices, num_slices, d_in, max_loras);

  cudaError_t precompute_err = cudaGetLastError();
  if (precompute_err != cudaSuccess) {
    return false;
  }

  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {total_problems, all_problems, nullptr},
    {(const cutlass_t**)ptr_X, stride_X, (const cutlass_t**)ptr_W, stride_W},
    {{1.0f, 0.0f}, (const cutlass_t**)ptr_Y, stride_C, ptr_Y, stride_Y},
    hw_info
  };

  Gemm gemm;
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    if (cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) {
      return false;
    }
  }

  if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  launch_scatter_add(y, y_sorted, token_indices_sorted, num_tokens, total_d_out, 
                     y_row_stride, y_sorted_stride, add_inputs, stream);

  if (workspace) cudaFreeAsync(workspace, stream);
  return true;
}

template <typename DType>
bool sgmv_shrink_vllm(DType *y, int64_t y_slice_stride,
                      DType *x_sorted,
                      DType **w,
                      int64_t w_lora_stride,
                      int32_t *lora_token_start_loc,
                      int32_t *active_lora_ids,
                      void *tmp_d,
                      int num_lora_indices,
                      int num_slices,
                      int num_tokens,
                      int d_in,
                      int d_out,
                      int max_loras,
                      cudaStream_t stream);

template <typename DType>
bool sgmv_expand_vllm(DType *y,
                      int64_t y_row_stride,
                      int64_t *slice_start_loc,
                      DType *x, int64_t x_slice_stride,
                      DType **w,
                      int64_t *w_lora_strides,
                      int32_t *lora_token_start_loc,
                      int32_t *active_lora_ids,
                      int32_t *token_indices_sorted,
                      int32_t *d_out_per_slice,
                      void *tmp_d,
                      DType *y_sorted,
                      int64_t y_sorted_stride,
                      int num_lora_indices,
                      int num_slices,
                      int num_tokens,
                      int d_in,
                      int max_loras,
                      bool add_inputs,
                      cudaStream_t stream);
