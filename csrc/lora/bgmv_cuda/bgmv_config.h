#pragma once

template <int feat_in, int feat_out, typename in_T, typename out_T,
          typename W_T>
void bgmv_kernel(out_T *__restrict__ Y, const in_T *__restrict__ X,
                 const W_T *__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size, int64_t num_layers,
                 int64_t layer_idx, float scale);

template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void bgmv_shrink_sliced(out_T *__restrict__ Y,
                        const in_T *__restrict__ X,
                        W_T **__restrict__ w_ptr,
                        const int64_t *__restrict__ indices,
                        int64_t num_tokens,
                        int64_t num_slices,
                        float scale);

template <int feat_in, int feat_out, typename in_T, typename out_T, typename W_T>
void bgmv_expand_sliced(out_T *__restrict__ Y,
                        const in_T *__restrict__ X,
                        W_T **__restrict__ w_ptr,
                        const int64_t *__restrict__ indices,
                        const int64_t *__restrict__ slice_start_loc,
                        int64_t num_tokens,
                        int64_t num_slices,
                        int64_t total_feat_out,
                        int32_t current_feat_out,
                        float scale);

// clang-format off

#define FOR_BGMV_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 128) \
    f(in_T, out_T, W_T, narrow, 256) \
    f(in_T, out_T, W_T, narrow, 512) \
    f(in_T, out_T, W_T, narrow, 640) \
    f(in_T, out_T, W_T, narrow, 896) \
    f(in_T, out_T, W_T, narrow, 1024) \
    f(in_T, out_T, W_T, narrow, 1280) \
    f(in_T, out_T, W_T, narrow, 1792) \
    f(in_T, out_T, W_T, narrow, 2048) \
    f(in_T, out_T, W_T, narrow, 2560) \
    f(in_T, out_T, W_T, narrow, 3584) \
    f(in_T, out_T, W_T, narrow, 4096) \
    f(in_T, out_T, W_T, narrow, 5120) \
    f(in_T, out_T, W_T, narrow, 7168) \
    f(in_T, out_T, W_T, narrow, 8192) \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 28672) \
    
    
// Keep above in sync with vllm/lora/layers::LogitsProcessorWithLoRA
// and vllm/tests/lora/test_punica.py

// Used for defining kernels going from the variety of
// dim in to the narrow dim out
    // Using it for the fully sharded column
    // parallel LoRA A which splits the rank dim
#define FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, 512, narrow) \
    f(in_T, out_T, W_T, 1024, narrow) \
    f(in_T, out_T, W_T, 2048, narrow) \
    f(in_T, out_T, W_T, 3584, narrow) \
    f(in_T, out_T, W_T, 4096, narrow) \
    f(in_T, out_T, W_T, 5120, narrow) \
    f(in_T, out_T, W_T, 7168, narrow) \
    f(in_T, out_T, W_T, 8192, narrow) \

// Keep above in sync with vllm/lora/layers::SamplerWithLoRA


// Keep this in sync with vllm/config::LoRAConfig
#define FOR_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 64)


#define FOR_INST_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 1) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 2) \
    FOR_INST_BGMV_NARROW(f, in_T, out_T, W_T, 4) \
    f(in_T, out_T, W_T, 8, 64) \
    f(in_T, out_T, W_T, 16, 64) \
    f(in_T, out_T, W_T, 32, 64) \
    f(in_T, out_T, W_T, 64, 64)

// clang-format on
