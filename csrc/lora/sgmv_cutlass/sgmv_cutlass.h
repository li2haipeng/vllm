#pragma once
#include <cuda_runtime.h>
#include <cstdint>

size_t sgmv_tmp_size_sliced(int num_loras, int num_slices);

// Multi-slice shrink (stacked weights, vLLM format)
// y: [num_slices, total_tokens, d_out], x: [total_tokens, d_in]
// w: [num_slices] pointers to stacked [num_loras, d_out, d_in] tensors
template <typename DType>
bool sgmv_shrink_stacked(DType *y, int64_t y_slice_stride,
                         DType *x,
                         DType **w,
                         int64_t w_lora_stride,
                         int32_t *s,
                         void *tmp_d,
                         int num_loras,
                         int num_slices,
                         int d_in,
                         int d_out,
                         cudaStream_t stream);

// Multi-slice expand (stacked weights, vLLM format)
// y: [total_tokens, sum(d_out_per_slice)] - 2D concatenated output (vLLM format)
// x: [num_slices, total_tokens, d_in]
// w: [num_slices] pointers to stacked [num_loras, d_out_i, d_in] tensors
// slice_start_loc: [num_slices] column offset for each slice in output
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
