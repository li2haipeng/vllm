// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "sgmv_cutlass.cuh"

template <>
bool sgmv_shrink_vllm<nv_half>(nv_half *y, int64_t y_slice_stride,
                               nv_half *x_sorted,
                               nv_half **w,
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
  return run_sgmv_shrink_kernel<ShrinkConfig>(
      y, y_slice_stride, x_sorted, w, w_lora_stride,
      lora_token_start_loc, active_lora_ids, tmp_d,
      num_lora_indices, num_slices, num_tokens, d_in, d_out, max_loras, stream);
}

template <>
bool sgmv_shrink_vllm<nv_bfloat16>(nv_bfloat16 *y, int64_t y_slice_stride,
                                   nv_bfloat16 *x_sorted,
                                   nv_bfloat16 **w,
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
  return run_sgmv_shrink_kernel<ShrinkConfigBf16>(
      y, y_slice_stride, x_sorted, w, w_lora_stride,
      lora_token_start_loc, active_lora_ids, tmp_d,
      num_lora_indices, num_slices, num_tokens, d_in, d_out, max_loras, stream);
}

template <>
bool sgmv_expand_vllm<nv_half>(nv_half *y,
                               int64_t y_row_stride,
                               int64_t *slice_start_loc,
                               nv_half *x, int64_t x_slice_stride,
                               nv_half **w,
                               int64_t *w_lora_strides,
                               int32_t *lora_token_start_loc,
                               int32_t *active_lora_ids,
                               int32_t *token_indices_sorted,
                               int32_t *d_out_per_slice,
                               void *tmp_d,
                               nv_half *y_sorted,
                               int64_t y_sorted_stride,
                               int num_lora_indices,
                               int num_slices,
                               int num_tokens,
                               int d_in,
                               int max_loras,
                               bool add_inputs,
                               cudaStream_t stream) {
  return run_sgmv_expand_kernel<ExpandConfig>(
      y, y_row_stride, slice_start_loc,
      x, x_slice_stride, w, w_lora_strides,
      lora_token_start_loc, active_lora_ids, token_indices_sorted,
      d_out_per_slice, tmp_d, y_sorted, y_sorted_stride,
      num_lora_indices, num_slices, num_tokens, d_in, max_loras, add_inputs, stream);
}

template <>
bool sgmv_expand_vllm<nv_bfloat16>(nv_bfloat16 *y,
                                   int64_t y_row_stride,
                                   int64_t *slice_start_loc,
                                   nv_bfloat16 *x, int64_t x_slice_stride,
                                   nv_bfloat16 **w,
                                   int64_t *w_lora_strides,
                                   int32_t *lora_token_start_loc,
                                   int32_t *active_lora_ids,
                                   int32_t *token_indices_sorted,
                                   int32_t *d_out_per_slice,
                                   void *tmp_d,
                                   nv_bfloat16 *y_sorted,
                                   int64_t y_sorted_stride,
                                   int num_lora_indices,
                                   int num_slices,
                                   int num_tokens,
                                   int d_in,
                                   int max_loras,
                                   bool add_inputs,
                                   cudaStream_t stream) {
  return run_sgmv_expand_kernel<ExpandConfigBf16>(
      y, y_row_stride, slice_start_loc,
      x, x_slice_stride, w, w_lora_strides,
      lora_token_start_loc, active_lora_ids, token_indices_sorted,
      d_out_per_slice, tmp_d, y_sorted, y_sorted_stride,
      num_lora_indices, num_slices, num_tokens, d_in, max_loras, add_inputs, stream);
}
