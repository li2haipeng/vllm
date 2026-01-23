#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

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
