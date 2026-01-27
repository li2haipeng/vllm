#include "core/registration.h"

#ifdef ENABLE_LORA_CUTLASS_SM90
#include "lora_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // y: [num_slices, num_tokens, d_out] output (zeroed inside kernel)
  // x_sorted: [num_tokens, d_in] input gathered by token_indices_sorted_by_lora_ids
  // w_list: list of weight tensors [num_slices], each [num_loras, d_out, d_in]
  // lora_token_start_loc: [num_lora_indices + 1] cumsum of tokens per lora group
  // active_lora_ids: [num_lora_indices] actual lora IDs for each group
  // w_ptr: pre-allocated buffer for weight pointers
  m.def("dispatch_sgmv_shrink_vllm(Tensor! y, Tensor x_sorted, Tensor[] w_list, "
        "Tensor lora_token_start_loc, Tensor active_lora_ids, Tensor tmp, Tensor w_ptr) -> ()");
  m.impl("dispatch_sgmv_shrink_vllm", torch::kCUDA, &dispatch_sgmv_shrink_vllm);

  // y: [num_tokens, total_d_out] output (scatter done internally)
  // x: [num_slices, num_tokens, d_in] input in sorted order (from shrink)
  // w_list: list of weight tensors [num_slices]
  // lora_token_start_loc: [num_lora_indices + 1]
  // active_lora_ids: [num_lora_indices]
  // d_out_per_slice: [num_slices] output dimension per slice
  // slice_start_loc: [num_slices] column offset for each slice
  // w_lora_strides: [num_slices] stride between loras per slice
  // token_indices_sorted: [num_tokens] indices for scatter
  // y_sorted: [num_tokens, total_d_out] pre-allocated buffer for sorted output
  // w_ptr: pre-allocated buffer for weight pointers
  // add_inputs: whether to add to existing y values
  m.def("dispatch_sgmv_expand_vllm(Tensor! y, Tensor x, Tensor[] w_list, "
        "Tensor lora_token_start_loc, Tensor active_lora_ids, "
        "Tensor d_out_per_slice, Tensor slice_start_loc, Tensor w_lora_strides, "
        "Tensor tmp, Tensor token_indices_sorted, Tensor y_sorted, Tensor w_ptr, bool add_inputs) -> ()");
  m.impl("dispatch_sgmv_expand_vllm", torch::kCUDA, &dispatch_sgmv_expand_vllm);

  // BGMV multi-slice shrink 
  // indices: [num_tokens] per-token lora mapping (indices[token_idx] -> lora_id)
  m.def(
      "dispatch_bgmv_shrink_sliced(Tensor! y, Tensor x, Tensor w_ptr, Tensor indices) -> ()");
  m.impl("dispatch_bgmv_shrink_sliced", torch::kCUDA, &dispatch_bgmv_shrink_sliced);

  // BGMV multi-slice expand
  // indices: [num_tokens] per-token lora mapping (indices[token_idx] -> lora_id)
  m.def(
      "dispatch_bgmv_expand_sliced(Tensor! y, Tensor x, Tensor w_ptr, Tensor indices, "
      "Tensor d_out_per_slice, Tensor slice_start_loc) -> ()");
  m.impl("dispatch_bgmv_expand_sliced", torch::kCUDA, &dispatch_bgmv_expand_sliced);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
