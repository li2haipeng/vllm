#pragma once

#include <torch/all.h>


void dispatch_sgmv_shrink_vllm(torch::Tensor y, torch::Tensor x_sorted,
                               torch::TensorList w_list,
                               torch::Tensor lora_token_start_loc,
                               torch::Tensor active_lora_ids,
                               torch::Tensor tmp,
                               torch::Tensor w_ptr);


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


void dispatch_bgmv_shrink_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices);


void dispatch_bgmv_expand_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices,
                                  torch::Tensor slice_start_loc,
                                  std::vector<int64_t> output_slices);
