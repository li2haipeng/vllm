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

// SM90 Hopper GEMM configuration using TMA + GMMA
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

size_t sgmv_tmp_size_sliced(int num_loras, int num_slices) {
  int total_problems = num_loras * num_slices;
  constexpr auto per_problem_sz = sizeof(void *) * 3 +
                                   sizeof(typename ExpandConfig::StrideA) * 4 +
                                   sizeof(typename ProblemShape::UnderlyingProblemShape);
  return per_problem_sz * total_problems + sizeof(int32_t) * num_slices * 2;
}

// Shrink stacked kernel args precomputation
template <typename GemmConfig, typename cutlass_t>
__global__ void precompute_sgmv_shrink_stacked_args(
    typename ProblemShape::UnderlyingProblemShape *all_problems,
    cutlass_t **ptr_y, cutlass_t **ptr_x, cutlass_t **ptr_w,
    typename GemmConfig::StrideA *stride_x,
    typename GemmConfig::StrideB *stride_w,
    typename GemmConfig::StrideC *stride_c,
    typename GemmConfig::StrideD *stride_y,
    cutlass_t *y, int64_t y_slice_stride,
    cutlass_t *x,
    cutlass_t **w,
    int64_t w_lora_stride,
    int32_t *s,
    int num_loras,
    int num_slices,
    int d_in,
    int d_out) {
  int problem_idx = blockIdx.x;
  int slice_id = problem_idx / num_loras;
  int lora_id = problem_idx % num_loras;

  int m = s[lora_id + 1] - s[lora_id];
  int k = d_in;
  int n = d_out;

  all_problems[problem_idx] = {m, n, k};
  ptr_w[problem_idx] = w[slice_id] + lora_id * w_lora_stride;
  ptr_x[problem_idx] = x + s[lora_id] * d_in;
  ptr_y[problem_idx] = y + slice_id * y_slice_stride + s[lora_id] * d_out;

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
bool run_sgmv_shrink_stacked_kernel(DType *y, int64_t y_slice_stride,
                                    DType *x,
                                    DType **w,
                                    int64_t w_lora_stride,
                                    int32_t *s,
                                    void *tmp_d,
                                    int num_loras,
                                    int num_slices,
                                    int num_tokens,
                                    int d_in,
                                    int d_out,
                                    cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  using Gemm = typename GemmConfig::Gemm;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  // Zero out y buffer - required for correctness
  size_t y_size_bytes = static_cast<size_t>(num_slices) * num_tokens * d_out * sizeof(DType);
  cudaMemsetAsync(y, 0, y_size_bytes, stream);

  int total_problems = num_loras * num_slices;

  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto stride_X = alloc_from_buf<StrideA>(&tmp_d, total_problems);
  auto stride_W = alloc_from_buf<StrideB>(&tmp_d, total_problems);
  auto stride_C = alloc_from_buf<StrideC>(&tmp_d, total_problems);
  auto stride_Y = alloc_from_buf<StrideD>(&tmp_d, total_problems);
  auto all_problems = alloc_from_buf<UnderlyingProblemShape>(&tmp_d, total_problems);

  precompute_sgmv_shrink_stacked_args<GemmConfig, cutlass_t><<<total_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, stride_X, stride_W, stride_C, stride_Y,
      (cutlass_t *)y, y_slice_stride,
      (cutlass_t *)x,
      (cutlass_t **)w,
      w_lora_stride,
      s, num_loras, num_slices, d_in, d_out);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {total_problems, all_problems, nullptr},
    {(const cutlass_t**)ptr_X, stride_X, (const cutlass_t**)ptr_W, stride_W},
    {{1.0f, 0.0f}, (const cutlass_t**)ptr_Y, stride_C, ptr_Y, stride_Y},
    hw_info
  };

  Gemm gemm;

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_shrink_stacked can_implement failed: %s\n",
            cutlassGetStatusString(status));
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t cuda_status = cudaMallocAsync(&workspace, workspace_size, stream);
    if (cuda_status != cudaSuccess) {
      fprintf(stderr, "sgmv_shrink_stacked workspace allocation failed: %s\n",
              cudaGetErrorString(cuda_status));
      return false;
    }
  }

  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_shrink_stacked initialize failed: %s\n",
            cutlassGetStatusString(status));
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_shrink_stacked run failed: %s\n",
            cutlassGetStatusString(status));
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  if (workspace) cudaFreeAsync(workspace, stream);
  return true;
}

// Expand stacked kernel args precomputation
// vLLM output format: [num_tokens, sum(d_out_per_slice)] - 2D concatenated tensor
// Each slice writes to a different column offset determined by slice_start_loc
template <typename GemmConfig, typename cutlass_t>
__global__ void precompute_sgmv_expand_stacked_args(
    typename ProblemShape::UnderlyingProblemShape *all_problems,
    cutlass_t **ptr_y, cutlass_t **ptr_x, cutlass_t **ptr_w,
    typename GemmConfig::StrideA *stride_x,
    typename GemmConfig::StrideB *stride_w,
    typename GemmConfig::StrideC *stride_c,
    typename GemmConfig::StrideD *stride_y,
    cutlass_t *y,
    int64_t y_row_stride,  // stride between rows (= total_d_out = sum of all d_out_per_slice)
    int64_t *slice_start_loc,  // column offset for each slice
    cutlass_t *x, int64_t x_slice_stride,
    cutlass_t **w,
    int64_t *w_lora_strides,
    int32_t *s,
    int32_t *d_out_per_slice,
    int num_loras,
    int num_slices,
    int d_in) {
  int problem_idx = blockIdx.x;
  int slice_id = problem_idx / num_loras;
  int lora_id = problem_idx % num_loras;

  int m = s[lora_id + 1] - s[lora_id];
  int k = d_in;
  int n = d_out_per_slice[slice_id];

  // Get column offset for this slice
  int64_t col_offset = slice_start_loc[slice_id];

  all_problems[problem_idx] = {m, n, k};
  ptr_w[problem_idx] = w[slice_id] + lora_id * w_lora_strides[slice_id];
  ptr_x[problem_idx] = x + slice_id * x_slice_stride + s[lora_id] * d_in;
  // Output: y[token_idx, col_offset:col_offset+n] for each token in segment
  // y pointer starts at row s[lora_id], column col_offset
  ptr_y[problem_idx] = y + s[lora_id] * y_row_stride + col_offset;

  stride_x[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideA{}, cute::make_shape(m, k, 1));
  stride_w[problem_idx] = cutlass::make_cute_packed_stride(
      typename GemmConfig::StrideB{}, cute::make_shape(n, k, 1));
  // Output stride: for row-major layout writing to a submatrix of a larger matrix,
  // we need the row stride to be y_row_stride (the leading dimension of the full output).
  // CUTLASS StrideC/StrideD for RowMajor is cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
  // which represents (row_stride, col_stride=1, batch_stride=0)
  // We create a stride with row_stride = y_row_stride so each row advances by total_d_out
  stride_c[problem_idx] = typename GemmConfig::StrideC{y_row_stride, cute::Int<1>{}, cute::Int<0>{}};
  stride_y[problem_idx] = typename GemmConfig::StrideD{y_row_stride, cute::Int<1>{}, cute::Int<0>{}};
}

template <typename GemmConfig, typename DType>
bool run_sgmv_expand_stacked_kernel(DType *y,
                                    int64_t y_row_stride,  // total_d_out = sum of all d_out_per_slice
                                    int64_t *slice_start_loc,  // column offset for each slice
                                    DType *x, int64_t x_slice_stride,
                                    DType **w,
                                    int64_t *w_lora_strides,
                                    int32_t *s,
                                    int32_t *d_out_per_slice,
                                    void *tmp_d,
                                    int num_loras,
                                    int num_slices,
                                    int d_in,
                                    cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  using Gemm = typename GemmConfig::Gemm;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  int total_problems = num_loras * num_slices;

  auto ptr_Y = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto ptr_W = alloc_from_buf<cutlass_t *>(&tmp_d, total_problems);
  auto stride_X = alloc_from_buf<StrideA>(&tmp_d, total_problems);
  auto stride_W = alloc_from_buf<StrideB>(&tmp_d, total_problems);
  auto stride_C = alloc_from_buf<StrideC>(&tmp_d, total_problems);
  auto stride_Y = alloc_from_buf<StrideD>(&tmp_d, total_problems);
  auto all_problems = alloc_from_buf<UnderlyingProblemShape>(&tmp_d, total_problems);

  precompute_sgmv_expand_stacked_args<GemmConfig, cutlass_t><<<total_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, stride_X, stride_W, stride_C, stride_Y,
      (cutlass_t *)y, y_row_stride, slice_start_loc,
      (cutlass_t *)x, x_slice_stride,
      (cutlass_t **)w,
      w_lora_strides,
      s, d_out_per_slice,
      num_loras, num_slices, d_in);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {total_problems, all_problems, nullptr},
    {(const cutlass_t**)ptr_X, stride_X, (const cutlass_t**)ptr_W, stride_W},
    {{1.0f, 0.0f}, (const cutlass_t**)ptr_Y, stride_C, ptr_Y, stride_Y},
    hw_info
  };

  Gemm gemm;

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_expand_stacked can_implement failed: %s\n",
            cutlassGetStatusString(status));
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    cudaError_t cuda_status = cudaMallocAsync(&workspace, workspace_size, stream);
    if (cuda_status != cudaSuccess) {
      fprintf(stderr, "sgmv_expand_stacked workspace allocation failed: %s\n",
              cudaGetErrorString(cuda_status));
      return false;
    }
  }

  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_expand_stacked initialize failed: %s\n",
            cutlassGetStatusString(status));
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "sgmv_expand_stacked run failed: %s\n",
            cutlassGetStatusString(status));
    if (workspace) cudaFreeAsync(workspace, stream);
    return false;
  }

  if (workspace) cudaFreeAsync(workspace, stream);
  return true;
}

// Template declarations
template <typename DType>
bool sgmv_shrink_stacked(DType *y, int64_t y_slice_stride,
                         DType *x,
                         DType **w,
                         int64_t w_lora_stride,
                         int32_t *s,
                         void *tmp_d,
                         int num_loras,
                         int num_slices,
                         int num_tokens,
                         int d_in,
                         int d_out,
                         cudaStream_t stream);

template <typename DType>
bool sgmv_expand_stacked(DType *y,
                         int64_t y_row_stride,  // total_d_out = sum of all d_out_per_slice
                         int64_t *slice_start_loc,  // column offset for each slice
                         DType *x, int64_t x_slice_stride,
                         DType **w,
                         int64_t *w_lora_strides,
                         int32_t *s,
                         int32_t *d_out_per_slice,
                         void *tmp_d,
                         int num_loras,
                         int num_slices,
                         int d_in,
                         cudaStream_t stream);
