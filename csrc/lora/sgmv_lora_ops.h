#pragma once

#include <torch/all.h>

// vLLM-compatible SGMV Shrink
// x_sorted: input gathered by token_indices_sorted_by_lora_ids
// lora_token_start_loc: [num_lora_indices + 1] cumsum of tokens per lora group
// active_lora_ids: [num_lora_indices] actual lora IDs for each group
// w_ptr: pre-allocated buffer for weight pointers (for cudagraph compatibility)
void dispatch_sgmv_shrink_vllm(torch::Tensor y, torch::Tensor x_sorted,
                               torch::TensorList w_list,
                               torch::Tensor lora_token_start_loc,
                               torch::Tensor active_lora_ids,
                               torch::Tensor tmp,
                               torch::Tensor w_ptr);

// vLLM-compatible SGMV Expand
// y: output tensor (scatter is done internally)
// x: input in sorted order (from shrink output)
// token_indices_sorted: indices for scatter operation
// y_sorted: pre-allocated buffer for intermediate sorted output
// w_ptr: pre-allocated buffer for weight pointers (for cudagraph compatibility)
// add_inputs: whether to add to existing y values
void dispatch_sgmv_expand_vllm(torch::Tensor y, torch::Tensor x,
                               torch::TensorList w_list,
                               torch::Tensor lora_token_start_loc,
                               torch::Tensor active_lora_ids,
                               torch::Tensor d_out_per_slice,
                               torch::Tensor slice_start_loc,
                               torch::Tensor w_lora_strides,
                               torch::Tensor tmp,
                               torch::Tensor token_indices_sorted,
                               torch::Tensor y_sorted,
                               torch::Tensor w_ptr,
                               bool add_inputs);
