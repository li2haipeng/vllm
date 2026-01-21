#include "core/registration.h"

#ifdef ENABLE_LORA_CUTLASS_SM90
#include "sgmv_lora_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Multi-slice SGMV Shrink (stacked weights - vLLM format)
  m.def("dispatch_sgmv_shrink_stacked(Tensor! y, Tensor x, Tensor w_ptr, Tensor s, "
        "Tensor tmp) -> ()");
  m.impl("dispatch_sgmv_shrink_stacked", torch::kCUDA, &dispatch_sgmv_shrink_stacked);

  // Multi-slice SGMV Expand (stacked weights - vLLM format)
  // y: [total_tokens, sum(d_out_per_slice)] - 2D concatenated output
  // x: [num_slices, total_tokens, d_in]
  // slice_start_loc: [num_slices] column offset for each slice in output
  m.def("dispatch_sgmv_expand_stacked(Tensor! y, Tensor x, Tensor w_ptr, Tensor s, "
        "Tensor d_out_per_slice, Tensor slice_start_loc, Tensor w_lora_strides, Tensor tmp) -> ()");
  m.impl("dispatch_sgmv_expand_stacked", torch::kCUDA, &dispatch_sgmv_expand_stacked);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
