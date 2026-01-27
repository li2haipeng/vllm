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

  TORCH_CHECK(lora_token_start_loc.scalar_type() == at::ScalarType::Int, 
              "lora_token_start_loc must be int32 tensor");
  TORCH_CHECK(active_lora_ids.scalar_type() == at::ScalarType::Int,
              "active_lora_ids must be int32 tensor");
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long,
              "w_ptr must be int64 tensor");
  TORCH_CHECK(w_ptr.size(0) >= num_slices, "w_ptr buffer too small");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x_sorted));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t* w_ptr_data = w_ptr.data_ptr<int64_t>();
  
  static thread_local torch::Tensor w_ptrs_cpu;
  if (!w_ptrs_cpu.defined() || w_ptrs_cpu.size(0) < num_slices) {
    w_ptrs_cpu = torch::empty({std::max(num_slices, 16)}, 
                               torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
  }
  
  int64_t* w_ptrs_host = w_ptrs_cpu.data_ptr<int64_t>();
  for (int i = 0; i < num_slices; i++) {
    CHECK_INPUT(w_list[i]);
    w_ptrs_host[i] = reinterpret_cast<int64_t>(w_list[i].data_ptr());
  }
  cudaMemcpyAsync(w_ptr_data, w_ptrs_host, num_slices * sizeof(int64_t), 
                  cudaMemcpyHostToDevice, stream);

  int64_t y_slice_stride = num_tokens * d_out;
  int64_t w_lora_stride = d_out * d_in;

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
        num_lora_indices, num_slices, num_tokens, d_in, d_out, stream);
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
        num_lora_indices, num_slices, num_tokens, d_in, d_out, stream);
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

  int64_t* w_ptr_data = w_ptr.data_ptr<int64_t>();
  
  static thread_local torch::Tensor w_ptrs_cpu_expand;
  if (!w_ptrs_cpu_expand.defined() || w_ptrs_cpu_expand.size(0) < num_slices) {
    w_ptrs_cpu_expand = torch::empty({std::max(num_slices, 16)}, 
                                      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
  }
  
  int64_t* w_ptrs_host = w_ptrs_cpu_expand.data_ptr<int64_t>();
  for (int i = 0; i < num_slices; i++) {
    CHECK_INPUT(w_list[i]);
    w_ptrs_host[i] = reinterpret_cast<int64_t>(w_list[i].data_ptr());
  }
  cudaMemcpyAsync(w_ptr_data, w_ptrs_host, num_slices * sizeof(int64_t), 
                  cudaMemcpyHostToDevice, stream);

  int64_t x_slice_stride = num_tokens * d_in;
  int64_t y_row_stride = total_d_out;
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
        num_lora_indices, num_slices, num_tokens, d_in, add_inputs, stream);
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
        num_lora_indices, num_slices, num_tokens, d_in, add_inputs, stream);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }

  TORCH_CHECK(ok, "CUTLASS SGMV expand kernel failed");
}

//================== BGMV Multi-Slice =================

// Multi-slice BGMV shrink kernel launcher (BGMV style with per-token indices)
template <typename T>
inline bool launch_bgmv_shrink_sliced_kernel(
    T *Y,                        // [num_slices, num_tokens, d_out]
    const T *X,                  // [num_tokens, d_in]
    T **w_ptr,                   // [num_slices] pointers to weight tensors
    const int64_t *indices,      // [num_tokens] per-token lora indices
    uint32_t d_in,               // hidden_size
    uint32_t d_out,              // lora_rank
    int64_t num_tokens,
    int64_t num_slices) {
  
  switch (pack_u32(d_in, d_out)) {
#define CASE_SHRINK_SLICED(_in_T, _out_T, _W_T, narrow, wide)                  \
  case pack_u32(wide, narrow):                                                 \
    bgmv_shrink_sliced<wide, narrow>(Y, X, w_ptr, indices, num_tokens,         \
                                     num_slices, 1.0f);                        \
    return true;

    FOR_BGMV_WIDE_NARROW(CASE_SHRINK_SLICED, T, T, T)
#undef CASE_SHRINK_SLICED
  default:
    return false;
  }
}

// Multi-slice BGMV expand kernel launcher (BGMV style with per-token indices)
// Supports variable d_out per slice (e.g., GQA with Q=4096, K=1024, V=1024)
template <typename T>
inline bool launch_bgmv_expand_sliced_kernel(
    T *Y,                            // [num_tokens, total_d_out]
    const T *X,                      // [num_slices, num_tokens, d_in]
    T **w_ptr,                       // [num_slices] pointers to weight tensors
    const int64_t *indices,          // [num_tokens] per-token lora indices
    const int32_t *d_out_per_slice,  // [num_slices] output dim per slice (host memory)
    const int64_t *slice_start_loc,  // [num_slices] column offset per slice
    uint32_t d_in,                   // lora_rank
    int64_t num_tokens,
    int64_t num_slices,
    int64_t total_d_out) {
  
  // Check if all slices have the same d_out (fast path)
  bool uniform_d_out = true;
  int32_t first_d_out = d_out_per_slice[0];
  for (int64_t i = 1; i < num_slices; ++i) {
    if (d_out_per_slice[i] != first_d_out) {
      uniform_d_out = false;
      break;
    }
  }
  
  if (uniform_d_out) {
    // Fast path: all slices have same d_out, launch single kernel for all slices
    switch (pack_u32(d_in, first_d_out)) {
#define CASE_EXPAND_SLICED(_in_T, _out_T, _W_T, narrow, wide)                  \
    case pack_u32(narrow, wide):                                               \
      bgmv_expand_sliced<narrow, wide>(Y, X, w_ptr, indices, slice_start_loc,  \
                                       num_tokens, num_slices,                 \
                                       total_d_out, wide, 1.0f);               \
      return true;

      FOR_BGMV_WIDE_NARROW(CASE_EXPAND_SLICED, T, T, T)
#undef CASE_EXPAND_SLICED
    default:
      return false;
    }
  } else {
    // Variable d_out path: launch separate kernel for each slice
    // Reuse bgmv_expand_sliced with num_slices=1 and offset pointers
    for (int64_t slice_id = 0; slice_id < num_slices; ++slice_id) {
      int32_t d_out = d_out_per_slice[slice_id];
      
      switch (pack_u32(d_in, d_out)) {
#define CASE_EXPAND_SINGLE(_in_T, _out_T, _W_T, narrow, wide)                  \
      case pack_u32(narrow, wide):                                             \
        bgmv_expand_sliced<narrow, wide>(                                      \
            Y,                                                                 \
            X + slice_id * num_tokens * d_in,                                  \
            w_ptr + slice_id,                                                  \
            indices,                                                           \
            slice_start_loc + slice_id,                                        \
            num_tokens, 1,                                                     \
            total_d_out, wide, 1.0f);                                          \
        break;

        FOR_BGMV_WIDE_NARROW(CASE_EXPAND_SINGLE, T, T, T)
#undef CASE_EXPAND_SINGLE
      default:
        return false;
      }
    }
    return true;
  }
}

void dispatch_bgmv_shrink_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(indices);
  
  CHECK_DIM(3, y);  // [num_slices, num_tokens, d_out]
  CHECK_DIM(2, x);  // [num_tokens, d_in]
  CHECK_DIM(1, w_ptr);  // [num_slices]
  CHECK_DIM(1, indices);  // [num_tokens] - per-token lora indices (BGMV style)
  
  int64_t num_slices = y.size(0);
  int64_t num_tokens = x.size(0);
  int64_t d_in = x.size(1);
  int64_t d_out = y.size(2);
  
  CHECK_EQ(y.size(1), num_tokens);
  CHECK_EQ(w_ptr.size(0), num_slices);
  CHECK_EQ(indices.size(0), num_tokens);
  
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long,
              "w_ptr must be int64 tensor");
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
              "indices must be int64 tensor (per-token lora mapping)");
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;
  
  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = launch_bgmv_shrink_sliced_kernel(
        static_cast<nv_half *>(y.data_ptr()),
        static_cast<nv_half *>(x.data_ptr()),
        reinterpret_cast<nv_half **>(w_ptr.data_ptr<int64_t>()),
        indices.data_ptr<int64_t>(),
        d_in, d_out, num_tokens, num_slices);
    break;
  case at::ScalarType::BFloat16:
    ok = launch_bgmv_shrink_sliced_kernel(
        static_cast<nv_bfloat16 *>(y.data_ptr()),
        static_cast<nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<nv_bfloat16 **>(w_ptr.data_ptr<int64_t>()),
        indices.data_ptr<int64_t>(),
        d_in, d_out, num_tokens, num_slices);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }
  
  TORCH_CHECK(ok, "BGMV shrink sliced kernel failed. d_in=", d_in, " d_out=", d_out);
}


void dispatch_bgmv_expand_sliced(torch::Tensor y, torch::Tensor x,
                                  torch::Tensor w_ptr, torch::Tensor indices,
                                  torch::Tensor d_out_per_slice,
                                  torch::Tensor slice_start_loc) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(indices);
  CHECK_INPUT(d_out_per_slice);
  CHECK_INPUT(slice_start_loc);
  
  CHECK_DIM(2, y);  // [num_tokens, total_d_out]
  CHECK_DIM(3, x);  // [num_slices, num_tokens, d_in]
  CHECK_DIM(1, w_ptr);  // [num_slices]
  CHECK_DIM(1, indices);  // [num_tokens] - per-token lora indices (BGMV style)
  CHECK_DIM(1, d_out_per_slice);  // [num_slices]
  CHECK_DIM(1, slice_start_loc);  // [num_slices]
  
  int64_t num_slices = x.size(0);
  int64_t num_tokens = x.size(1);
  int64_t d_in = x.size(2);
  int64_t total_d_out = y.size(1);
  
  CHECK_EQ(y.size(0), num_tokens);
  CHECK_EQ(w_ptr.size(0), num_slices);
  CHECK_EQ(indices.size(0), num_tokens);
  CHECK_EQ(d_out_per_slice.size(0), num_slices);
  CHECK_EQ(slice_start_loc.size(0), num_slices);
  
  TORCH_CHECK(w_ptr.scalar_type() == at::ScalarType::Long,
              "w_ptr must be int64 tensor");
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
              "indices must be int64 tensor (per-token lora mapping)");
  TORCH_CHECK(d_out_per_slice.scalar_type() == at::ScalarType::Int,
              "d_out_per_slice must be int32 tensor");
  TORCH_CHECK(slice_start_loc.scalar_type() == at::ScalarType::Long,
              "slice_start_loc must be int64 tensor");
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;
  
  // Copy d_out_per_slice to host for kernel dispatch
  std::vector<int32_t> d_out_host(num_slices);
  cudaMemcpy(d_out_host.data(), d_out_per_slice.data_ptr<int32_t>(), 
             num_slices * sizeof(int32_t), cudaMemcpyDeviceToHost);
  
  switch (x.scalar_type()) {
  case at::ScalarType::Half:
    ok = launch_bgmv_expand_sliced_kernel(
        static_cast<nv_half *>(y.data_ptr()),
        static_cast<nv_half *>(x.data_ptr()),
        reinterpret_cast<nv_half **>(w_ptr.data_ptr<int64_t>()),
        indices.data_ptr<int64_t>(),
        d_out_host.data(),
        slice_start_loc.data_ptr<int64_t>(),
        d_in, num_tokens, num_slices, total_d_out);
    break;
  case at::ScalarType::BFloat16:
    ok = launch_bgmv_expand_sliced_kernel(
        static_cast<nv_bfloat16 *>(y.data_ptr()),
        static_cast<nv_bfloat16 *>(x.data_ptr()),
        reinterpret_cast<nv_bfloat16 **>(w_ptr.data_ptr<int64_t>()),
        indices.data_ptr<int64_t>(),
        d_out_host.data(),
        slice_start_loc.data_ptr<int64_t>(),
        d_in, num_tokens, num_slices, total_d_out);
    break;
  default:
    TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
  }
  
  TORCH_CHECK(ok, "BGMV expand sliced kernel failed. d_in=", d_in, " total_d_out=", total_d_out);
}