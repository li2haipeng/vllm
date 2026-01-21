#pragma once

#include <torch/all.h>

// Multi-slice shrink (stacked weights - vLLM format)
void dispatch_sgmv_shrink_stacked(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor s,
                                  torch::Tensor tmp);

// Multi-slice expand (stacked weights - vLLM format)
// y: [total_tokens, sum(d_out_per_slice)] - 2D concatenated output
// x: [num_slices, total_tokens, d_in]
// slice_start_loc: [num_slices] column offset for each slice in output
// w_lora_strides: [num_slices] stride between loras per slice (= d_out_per_slice[i] * d_in)
void dispatch_sgmv_expand_stacked(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor s,
                                  torch::Tensor d_out_per_slice,
                                  torch::Tensor slice_start_loc,
                                  torch::Tensor w_lora_strides,
                                  torch::Tensor tmp);
