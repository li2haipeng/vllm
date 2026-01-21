#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "sgmv_cutlass/sgmv_cutlass.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

void dispatch_sgmv_shrink_stacked(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor s,
                                  torch::Tensor tmp) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(3, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  int num_slices = y.size(0);
  int total_tokens = y.size(1);
  int d_out = y.size(2);
  int d_in = x.size(1);
  int num_loras = s.size(0) - 1;

  CHECK_EQ(x.size(0), total_tokens);
  CHECK_EQ(w_ptr.size(0), num_slices);

  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long, "w_ptr must be int64 tensor");
  TORCH_CHECK(s.scalar_type() == at::ScalarType::Int, "s must be int32 tensor");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t y_slice_stride = total_tokens * d_out;
  int64_t w_lora_stride = d_out * d_in;

  bool ok = false;

  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = sgmv_shrink_stacked(
        static_cast<nv_half*>(y.data_ptr()),
        y_slice_stride,
        static_cast<nv_half*>(x.data_ptr()),
        reinterpret_cast<nv_half**>(w_ptr.data_ptr<int64_t>()),
        w_lora_stride,
        s.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_loras, num_slices, d_in, d_out, stream);
    break;
  case at::ScalarType::BFloat16:
    ok = sgmv_shrink_stacked(
        static_cast<nv_bfloat16*>(y.data_ptr()),
        y_slice_stride,
        static_cast<nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<nv_bfloat16**>(w_ptr.data_ptr<int64_t>()),
        w_lora_stride,
        s.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_loras, num_slices, d_in, d_out, stream);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "CUTLASS SGMV shrink stacked kernel failed");
}

void dispatch_sgmv_expand_stacked(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor s,
                                  torch::Tensor d_out_per_slice,
                                  torch::Tensor slice_start_loc,
                                  torch::Tensor w_lora_strides,
                                  torch::Tensor tmp) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(d_out_per_slice);
  CHECK_INPUT(slice_start_loc);
  CHECK_INPUT(w_lora_strides);
  CHECK_INPUT(tmp);

  // vLLM format: y is 2D [total_tokens, total_d_out], x is 3D [num_slices, total_tokens, d_in]
  CHECK_DIM(2, y);
  CHECK_DIM(3, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, d_out_per_slice);
  CHECK_DIM(1, slice_start_loc);
  CHECK_DIM(1, w_lora_strides);
  CHECK_DIM(1, tmp);

  int num_slices = x.size(0);
  int total_tokens = x.size(1);
  int d_in = x.size(2);
  int num_loras = s.size(0) - 1;
  int total_d_out = y.size(1);  // sum of all d_out_per_slice

  CHECK_EQ(y.size(0), total_tokens);
  CHECK_EQ(w_ptr.size(0), num_slices);
  CHECK_EQ(d_out_per_slice.size(0), num_slices);
  CHECK_EQ(slice_start_loc.size(0), num_slices);
  CHECK_EQ(w_lora_strides.size(0), num_slices);

  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long, "w_ptr must be int64 tensor");
  TORCH_CHECK(s.scalar_type() == at::ScalarType::Int, "s must be int32 tensor");
  TORCH_CHECK(d_out_per_slice.scalar_type() == at::ScalarType::Int, "d_out_per_slice must be int32 tensor");
  TORCH_CHECK(slice_start_loc.scalar_type() == at::ScalarType::Long, "slice_start_loc must be int64 tensor");
  TORCH_CHECK(w_lora_strides.scalar_type() == at::ScalarType::Long, "w_lora_strides must be int64 tensor");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t x_slice_stride = total_tokens * d_in;
  int64_t y_row_stride = total_d_out;  // stride between rows in output

  bool ok = false;

  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = sgmv_expand_stacked(
        static_cast<nv_half*>(y.data_ptr()),
        y_row_stride,
        slice_start_loc.data_ptr<int64_t>(),
        static_cast<nv_half*>(x.data_ptr()),
        x_slice_stride,
        reinterpret_cast<nv_half**>(w_ptr.data_ptr<int64_t>()),
        w_lora_strides.data_ptr<int64_t>(),
        s.data_ptr<int32_t>(),
        d_out_per_slice.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_loras, num_slices, d_in, stream);
    break;
  case at::ScalarType::BFloat16:
    ok = sgmv_expand_stacked(
        static_cast<nv_bfloat16*>(y.data_ptr()),
        y_row_stride,
        slice_start_loc.data_ptr<int64_t>(),
        static_cast<nv_bfloat16*>(x.data_ptr()),
        x_slice_stride,
        reinterpret_cast<nv_bfloat16**>(w_ptr.data_ptr<int64_t>()),
        w_lora_strides.data_ptr<int64_t>(),
        s.data_ptr<int32_t>(),
        d_out_per_slice.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_loras, num_slices, d_in, stream);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "CUTLASS SGMV expand stacked kernel failed");
}
