#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// vLLM-compatible SGMV Shrink kernel
// Input x_sorted should be pre-gathered using token_indices_sorted_by_lora_ids
// Output y is in sorted order (same order as x_sorted)
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
                      cudaStream_t stream);

// vLLM-compatible SGMV Expand kernel
// Input x is in sorted order (output from shrink)
// Output is scattered to y using token_indices_sorted
// y_sorted is a pre-allocated buffer for intermediate sorted output
// y_sorted_stride is the row stride of y_sorted (may differ from y_row_stride if y_sorted is a view)
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
                      bool add_inputs,
                      cudaStream_t stream);
