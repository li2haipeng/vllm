// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

#include "sgmv_cutlass/sgmv_cutlass.h"
#include "bgmv_cuda/bgmv_config.h"
#include "bgmv_cuda/bgmv_impl.cuh"

inline constexpr uint64_t pack_u32(uint32_t a, uint32_t b) {
  return (uint64_t(a) << 32) | uint64_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

void dispatch_sgmv_shrink_vllm(torch::Tensor y, torch::Tensor x_sorted,
                               torch::TensorList w_list,
                               torch::Tensor lora_token_start_loc,
                               torch::Tensor active_lora_ids,
                               torch::Tensor tmp,
                               torch::Tensor w_ptr) {
  CHECK_INPUT(y);
  CHECK_INPUT(x_sorted);
  CHECK_INPUT(lora_token_start_loc);
  CHECK_INPUT(active_lora_ids);
  CHECK_INPUT(tmp);
  CHECK_INPUT(w_ptr);

  CHECK_DIM(3, y);
  CHECK_DIM(2, x_sorted);
  CHECK_DIM(1, lora_token_start_loc);
  CHECK_DIM(1, active_lora_ids);
  CHECK_DIM(1, tmp);
  CHECK_DIM(1, w_ptr);

  int num_slices = y.size(0);
  int num_tokens = y.size(1);
  int d_out = y.size(2);
  int d_in = x_sorted.size(1);
  int num_lora_indices = active_lora_ids.size(0);

  CHECK_EQ(x_sorted.size(0), num_tokens);
  CHECK_EQ(static_cast<int>(w_list.size()), num_slices);
  CHECK_EQ(lora_token_start_loc.size(0), num_lora_indices + 1);

  // Weight tensor shape: [max_loras, 1, d_out, d_in]
  int max_loras = 0;
  for (int i = 0; i < num_slices; i++) {
    TORCH_CHECK(w_list[i].dim() == 4, 
                "Weight tensor must be 4D [max_loras, 1, d_out, d_in], got ", w_list[i].dim(), "D");
    TORCH_CHECK(w_list[i].size(2) == d_out,
                "Weight d_out mismatch: expected ", d_out, ", got ", w_list[i].size(2));
    TORCH_CHECK(w_list[i].size(3) == d_in,
                "Weight d_in mismatch: expected ", d_in, ", got ", w_list[i].size(3));
    TORCH_CHECK(w_list[i].device() == x_sorted.device(),
                "Weight tensor device mismatch");
    if (i == 0) {
      max_loras = w_list[i].size(0);
    } else {
      TORCH_CHECK(w_list[i].size(0) == max_loras,
                  "All weight tensors must have same max_loras dimension");
    }
  }

  TORCH_CHECK(lora_token_start_loc.scalar_type() == at::ScalarType::Int, 
              "lora_token_start_loc must be int32 tensor");
  TORCH_CHECK(active_lora_ids.scalar_type() == at::ScalarType::Int,
              "active_lora_ids must be int32 tensor");
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long,
              "w_ptr must be int64 tensor");
  TORCH_CHECK(w_ptr.size(0) >= num_slices, "w_ptr buffer too small");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x_sorted));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  
  int device_id;
  cudaGetDevice(&device_id);

  int64_t* w_ptr_data = w_ptr.data_ptr<int64_t>();
  
  std::vector<int64_t> w_ptrs_host(num_slices);
  for (int i = 0; i < num_slices; i++) {
    CHECK_INPUT(w_list[i]);
    TORCH_CHECK(w_list[i].device().index() == device_id,
                "Weight tensor on wrong device");
    w_ptrs_host[i] = reinterpret_cast<int64_t>(w_list[i].data_ptr());
  }
  cudaMemcpy(w_ptr_data, w_ptrs_host.data(), num_slices * sizeof(int64_t), cudaMemcpyHostToDevice);

  int64_t y_slice_stride = num_tokens * d_out;
  int64_t w_lora_stride = w_list[0].stride(0);

  bool ok = false;

  switch (x_sorted.scalar_type()) {
  case at::ScalarType::Half:
    ok = sgmv_shrink_vllm(
        static_cast<nv_half*>(y.data_ptr()),
        y_slice_stride,
        static_cast<nv_half*>(x_sorted.data_ptr()),
        reinterpret_cast<nv_half**>(w_ptr_data),
        w_lora_stride,
        lora_token_start_loc.data_ptr<int32_t>(),
        active_lora_ids.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_lora_indices, num_slices, num_tokens, d_in, d_out, max_loras, stream);
    break;
  case at::ScalarType::BFloat16:
    ok = sgmv_shrink_vllm(
        static_cast<nv_bfloat16*>(y.data_ptr()),
        y_slice_stride,
        static_cast<nv_bfloat16*>(x_sorted.data_ptr()),
        reinterpret_cast<nv_bfloat16**>(w_ptr_data),
        w_lora_stride,
        lora_token_start_loc.data_ptr<int32_t>(),
        active_lora_ids.data_ptr<int32_t>(),
        tmp.data_ptr(),
        num_lora_indices, num_slices, num_tokens, d_in, d_out, max_loras, stream);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x_sorted.scalar_type());
  }

  TORCH_CHECK(ok, "CUTLASS SGMV shrink kernel failed");
}

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
                               bool add_inputs) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(lora_token_start_loc);
  CHECK_INPUT(active_lora_ids);
  CHECK_INPUT(d_out_per_slice);
  CHECK_INPUT(slice_start_loc);
  CHECK_INPUT(w_lora_strides);
  CHECK_INPUT(tmp);
  CHECK_INPUT(token_indices_sorted);
  CHECK_CUDA(y_sorted);
  CHECK_INPUT(w_ptr);

  CHECK_DIM(2, y);
  CHECK_DIM(3, x);
  CHECK_DIM(1, lora_token_start_loc);
  CHECK_DIM(1, active_lora_ids);
  CHECK_DIM(1, d_out_per_slice);
  CHECK_DIM(1, slice_start_loc);
  CHECK_DIM(1, w_lora_strides);
  CHECK_DIM(1, tmp);
  CHECK_DIM(1, token_indices_sorted);
  CHECK_DIM(2, y_sorted);
  CHECK_DIM(1, w_ptr);

  int num_slices = x.size(0);
  int num_tokens = x.size(1);
  int d_in = x.size(2);
  int num_lora_indices = active_lora_ids.size(0);
  int total_d_out = y.size(1);

  CHECK_EQ(y.size(0), num_tokens);
  CHECK_EQ(static_cast<int>(w_list.size()), num_slices);
  CHECK_EQ(lora_token_start_loc.size(0), num_lora_indices + 1);
  CHECK_EQ(y_sorted.size(0), num_tokens);
  CHECK_EQ(y_sorted.size(1), total_d_out);

  // Weight tensor shape: [max_loras, 1, d_out_slice, d_in]
  int max_loras = 0;
  for (int i = 0; i < num_slices; i++) {
    TORCH_CHECK(w_list[i].dim() == 4, 
                "Weight tensor must be 4D [max_loras, 1, d_out, d_in], got ", w_list[i].dim(), "D");
    TORCH_CHECK(w_list[i].size(3) == d_in,
                "Weight d_in mismatch: expected ", d_in, ", got ", w_list[i].size(3));
    TORCH_CHECK(w_list[i].device() == x.device(),
                "Weight tensor device mismatch");
    if (i == 0) {
      max_loras = w_list[i].size(0);
    } else {
      TORCH_CHECK(w_list[i].size(0) == max_loras,
                  "All weight tensors must have same max_loras dimension");
    }
  }

  TORCH_CHECK(lora_token_start_loc.scalar_type() == at::ScalarType::Int,
              "lora_token_start_loc must be int32 tensor");
  TORCH_CHECK(active_lora_ids.scalar_type() == at::ScalarType::Int,
              "active_lora_ids must be int32 tensor");
  TORCH_CHECK(d_out_per_slice.scalar_type() == at::ScalarType::Int,
              "d_out_per_slice must be int32 tensor");
  TORCH_CHECK(slice_start_loc.scalar_type() == at::ScalarType::Long,
              "slice_start_loc must be int64 tensor");
  TORCH_CHECK(w_lora_strides.scalar_type() == at::ScalarType::Long,
              "w_lora_strides must be int64 tensor");
  TORCH_CHECK(y_sorted.scalar_type() == x.scalar_type(),
              "y_sorted must have same dtype as x");
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long,
              "w_ptr must be int64 tensor");
  TORCH_CHECK(w_ptr.size(0) >= num_slices, "w_ptr buffer too small");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  
  int device_id;
  cudaGetDevice(&device_id);

  int64_t* w_ptr_data = w_ptr.data_ptr<int64_t>();
  
  std::vector<int64_t> w_ptrs_host(num_slices);
  for (int i = 0; i < num_slices; i++) {
    CHECK_INPUT(w_list[i]);
    TORCH_CHECK(w_list[i].device().index() == device_id,
                "Weight tensor on wrong device");
    w_ptrs_host[i] = reinterpret_cast<int64_t>(w_list[i].data_ptr());
  }
  cudaMemcpy(w_ptr_data, w_ptrs_host.data(), num_slices * sizeof(int64_t), cudaMemcpyHostToDevice);

  int64_t x_slice_stride = num_tokens * d_in;
  int64_t y_row_stride = y.stride(0);
  int64_t y_sorted_stride = y_sorted.stride(0);

  bool ok = false;

  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = sgmv_expand_vllm(
        static_cast<nv_half*>(y.data_ptr()),
        y_row_stride,
        slice_start_loc.data_ptr<int64_t>(),
        static_cast<nv_half*>(x.data_ptr()),
        x_slice_stride,
        reinterpret_cast<nv_half**>(w_ptr_data),
        w_lora_strides.data_ptr<int64_t>(),
        lora_token_start_loc.data_ptr<int32_t>(),
        active_lora_ids.data_ptr<int32_t>(),
        token_indices_sorted.data_ptr<int32_t>(),
        d_out_per_slice.data_ptr<int32_t>(),
        tmp.data_ptr(),
        static_cast<nv_half*>(y_sorted.data_ptr()),
        y_sorted_stride,
        num_lora_indices, num_slices, num_tokens, d_in, max_loras, add_inputs, stream);
    break;
  case at::ScalarType::BFloat16:
    ok = sgmv_expand_vllm(
        static_cast<nv_bfloat16*>(y.data_ptr()),
        y_row_stride,
        slice_start_loc.data_ptr<int64_t>(),
        static_cast<nv_bfloat16*>(x.data_ptr()),
        x_slice_stride,
        reinterpret_cast<nv_bfloat16**>(w_ptr_data),
        w_lora_strides.data_ptr<int64_t>(),
        lora_token_start_loc.data_ptr<int32_t>(),
        active_lora_ids.data_ptr<int32_t>(),
        token_indices_sorted.data_ptr<int32_t>(),
        d_out_per_slice.data_ptr<int32_t>(),
        tmp.data_ptr(),
        static_cast<nv_bfloat16*>(y_sorted.data_ptr()),
        y_sorted_stride,
        num_lora_indices, num_slices, num_tokens, d_in, max_loras, add_inputs, stream);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "CUTLASS SGMV expand kernel failed");
}

// BGMV sliced stubs - not implemented, use SGMV kernels instead
void dispatch_bgmv_shrink_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices) {
  TORCH_CHECK(false, "BGMV sliced kernels not implemented. Use SGMV kernels.");
}

void dispatch_bgmv_expand_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices,
                                  torch::Tensor d_out_per_slice,
                                  torch::Tensor slice_start_loc) {
  TORCH_CHECK(false, "BGMV sliced kernels not implemented. Use SGMV kernels.");
}
