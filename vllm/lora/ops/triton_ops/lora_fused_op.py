# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

import torch

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_expand_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr, _get_lora_b_ptr, get_v1_op_configs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op
from typing import Optional

@triton.jit
def _lora_fused_kernel(
    input_ptr,
    lora_a_ptr,
    lora_b_ptr,
    output_ptr,
    N,
    R,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    ls_a_d0_ptr,
    ls_a_d1_ptr,
    ls_a_d2_ptr,
    ls_b_d0_ptr,
    ls_b_d1_ptr,
    ls_b_d2_ptr,
    output_d0_stride,
    output_d1_stride,  # 1
    output_hs_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    # if pid_m == 0:
    #     print("cta_m_len", cta_m_len)
    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)
    # print("cta_lora_seq_indices", cta_lora_seq_indices)
    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_expand_kernel(
        lora_id,
        slice_id,
        input_ptr,
        lora_a_ptr,
        lora_b_ptr,
        output_ptr,
        curr_N,
        R,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        slice_start_loc,
        # input ptr strides
        input_d0_stride,
        input_d1_stride,
        # lora ptr strides
        ls_a_d0_ptr,
        ls_a_d1_ptr,
        ls_a_d2_ptr,
        ls_b_d0_ptr,
        ls_b_d1_ptr,
        ls_b_d2_ptr,
        # out ptr strides
        output_d0_stride,
        output_d1_stride,
        # constants
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_R,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        ADD_INPUTS
    )


@torch.inference_mode()
def _lora_shrink_expand(
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    lora_b_weights: list[torch.Tensor],  # shape [num_loras, hidden_size, lora_rank]
    output_tensor: torch.Tensor,  # shape [num_tokens, hidden_size * num_slices]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens] 
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    scaling: float,
    no_lora_flag_cpu: torch.Tensor,  # shape [1] 
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    assert len(lora_a_weights) == len(lora_b_weights)
    # metadata sanity check
    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(
        0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (lora_a_ptr_tensor, lora_a_strides_d0, lora_a_strides_d1,
     lora_a_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    
    (slice_start_tensor, lora_b_ptr_tensor, lora_b_strides_d0,
     lora_b_strides_d1, lora_b_strides_d2, hidden_sizes_tensor,
     same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start,
                                           inputs.device)

    R = lora_b_weights[0].shape[-1]  # K= rank
    ADD_INPUTS = add_inputs
    N, K = lora_a_weights[0].shape[-2:]  # K=hidden_size,N=rank
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    kernel_config = get_v1_op_configs(op_type="fused",
                                    max_loras=MAX_LORAS,
                                    batch=M,
                                    hidden_size=K,
                                    rank=N,
                                    num_slices=NUM_SLICES,
                                    add_inputs=add_inputs)

    BLOCK_M = kernel_config['block_m']
    BLOCK_N = kernel_config['block_n']
    BLOCK_K = kernel_config['block_k']
    BLOCK_R = kernel_config['block_r']
    NUM_WARPS = kernel_config['num_warps']
    NUM_STAGES = kernel_config['num_stages']

    EVEN_K = K % BLOCK_K == 0  # type: ignore
    EVEN_R = R % BLOCK_K == 0  # type: ignore

    # TODO (varun): This grid formulation maximizes parallelization at the
    # cost of wasteful thread block launch when only a few input tokens require
    # LoRA. This might not be the best in all cases.
    grid = (
        triton.cdiv(M, BLOCK_M),
        NUM_SLICES,
        # Each LoRA receives its own set of thread blocks for output
        # computation. If some LoRA doesn't have any tokens to process, its
        # thread blocks simply exit.
        MAX_LORAS,
    )

    _lora_fused_kernel[grid](
        inputs,
        lora_a_ptr_tensor,
        lora_b_ptr_tensor,
        output_tensor,
        MAX_N,
        R,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_a_strides_d0,
        lora_a_strides_d1,
        lora_a_strides_d2,
        lora_b_strides_d0,
        lora_b_strides_d1,
        lora_b_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_R,
        EVEN_K,
        ADD_INPUTS,
        NUM_SLICES,
        same_stride,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return

def _lora_shrink_expand_fake(
    inputs: torch.Tensor, 
    lora_a_weights: list[torch.Tensor], 
    lora_b_weights: list[torch.Tensor],  
    output_tensor: torch.Tensor,  
    token_lora_mapping: torch.Tensor, 
    token_indices_sorted_by_lora_ids: torch.Tensor, 
    num_tokens_per_lora: torch.Tensor,  
    lora_token_start_loc: torch.Tensor,  
    lora_ids: torch.Tensor, 
    scaling: float,
    no_lora_flag_cpu: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_shrink_expand",
        op_func=_lora_shrink_expand,
        mutates_args=["output_tensor"],
        fake_impl=_lora_shrink_expand_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    lora_shrink_expand = torch.ops.vllm.lora_shrink_expand

except AttributeError:
    fused_shrink_expand = _lora_shrink_expand
