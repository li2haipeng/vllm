#include "core/registration.h"

#ifdef ENABLE_LORA_CUTLASS_SM90
#include "lora_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {

  m.def("dispatch_sgmv_shrink_vllm(Tensor! y, Tensor x_sorted, Tensor[] w_list, "
        "Tensor lora_token_start_loc, Tensor active_lora_ids, Tensor tmp, Tensor w_ptr) -> ()");
  m.impl("dispatch_sgmv_shrink_vllm", torch::kCUDA, &dispatch_sgmv_shrink_vllm);


  m.def("dispatch_sgmv_expand_vllm(Tensor! y, Tensor x, Tensor[] w_list, "
        "Tensor lora_token_start_loc, Tensor active_lora_ids, "
        "Tensor d_out_per_slice, Tensor slice_start_loc, Tensor w_lora_strides, "
        "Tensor tmp, Tensor token_indices_sorted, Tensor y_sorted, Tensor w_ptr, bool add_inputs) -> ()");
  m.impl("dispatch_sgmv_expand_vllm", torch::kCUDA, &dispatch_sgmv_expand_vllm);


  m.def(
      "dispatch_bgmv_shrink_sliced(Tensor! y, Tensor x, Tensor w_ptr, Tensor indices) -> ()");
  m.impl("dispatch_bgmv_shrink_sliced", torch::kCUDA, &dispatch_bgmv_shrink_sliced);


  m.def(
      "dispatch_bgmv_expand_sliced(Tensor! y, Tensor x, Tensor w_ptr, Tensor indices, "
      "Tensor slice_start_loc, int[] output_slices) -> ()");
  m.impl("dispatch_bgmv_expand_sliced", torch::kCUDA, &dispatch_bgmv_expand_sliced);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
