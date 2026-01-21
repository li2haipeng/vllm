#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "sgmv_cutlass.cuh"

// Multi-slice shrink stacked
template <>
bool sgmv_shrink_stacked<half>(half *y, int64_t y_slice_stride,
                               half *x,
                               half **w,
                               int64_t w_lora_stride,
                               int32_t *s,
                               void *tmp_d,
                               int num_loras,
                               int num_slices,
                               int d_in,
                               int d_out,
                               cudaStream_t stream) {
  return run_sgmv_shrink_stacked_kernel<ShrinkConfig>(
      y, y_slice_stride, x, w, w_lora_stride, s, tmp_d,
      num_loras, num_slices, d_in, d_out, stream);
}

template <>
bool sgmv_shrink_stacked<nv_bfloat16>(nv_bfloat16 *y, int64_t y_slice_stride,
                                      nv_bfloat16 *x,
                                      nv_bfloat16 **w,
                                      int64_t w_lora_stride,
                                      int32_t *s,
                                      void *tmp_d,
                                      int num_loras,
                                      int num_slices,
                                      int d_in,
                                      int d_out,
                                      cudaStream_t stream) {
  return run_sgmv_shrink_stacked_kernel<ShrinkConfigBf16>(
      y, y_slice_stride, x, w, w_lora_stride, s, tmp_d,
      num_loras, num_slices, d_in, d_out, stream);
}

// Multi-slice expand stacked
// Output: [total_tokens, sum(d_out_per_slice)] - 2D concatenated tensor
template <>
bool sgmv_expand_stacked<half>(half *y,
                               int64_t y_row_stride,
                               int64_t *slice_start_loc,
                               half *x, int64_t x_slice_stride,
                               half **w,
                               int64_t *w_lora_strides,
                               int32_t *s,
                               int32_t *d_out_per_slice,
                               void *tmp_d,
                               int num_loras,
                               int num_slices,
                               int d_in,
                               cudaStream_t stream) {
  return run_sgmv_expand_stacked_kernel<ExpandConfig>(
      y, y_row_stride, slice_start_loc, x, x_slice_stride, w, w_lora_strides,
      s, d_out_per_slice, tmp_d, num_loras, num_slices, d_in, stream);
}

template <>
bool sgmv_expand_stacked<nv_bfloat16>(nv_bfloat16 *y,
                                      int64_t y_row_stride,
                                      int64_t *slice_start_loc,
                                      nv_bfloat16 *x, int64_t x_slice_stride,
                                      nv_bfloat16 **w,
                                      int64_t *w_lora_strides,
                                      int32_t *s,
                                      int32_t *d_out_per_slice,
                                      void *tmp_d,
                                      int num_loras,
                                      int num_slices,
                                      int d_in,
                                      cudaStream_t stream) {
  return run_sgmv_expand_stacked_kernel<ExpandConfigBf16>(
      y, y_row_stride, slice_start_loc, x, x_slice_stride, w, w_lora_strides,
      s, d_out_per_slice, tmp_d, num_loras, num_slices, d_in, stream);
}
