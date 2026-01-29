# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS SGMV and BGMV LoRA kernel Python wrappers."""

import torch

import vllm._lora_C  # noqa: F401
from vllm.utils.torch_utils import direct_register_custom_op


@torch.inference_mode()
def _cutlass_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    token_indices_sorted_int64: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    scale: float,
) -> None:
    if no_lora_flag_cpu.item():
        return

    num_tokens = x.size(0)
    x_sorted = torch.index_select(x, 0,
                                  token_indices_sorted_int64[:num_tokens])

    torch.ops._lora_C.dispatch_sgmv_shrink_vllm(y, x_sorted, lora_a_weights,
                                                lora_token_start_loc, lora_ids,
                                                cutlass_tmp, w_ptr_buffer)


def _cutlass_shrink_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    token_indices_sorted_int64: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    scale: float,
) -> None:
    return


@torch.inference_mode()
def _cutlass_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_slices: list[int],
    offset_start: int,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    y_sorted_buffer: torch.Tensor,
    d_out_per_slice_buffer: torch.Tensor,
    slice_start_loc_buffer: torch.Tensor,
    w_lora_strides_buffer: torch.Tensor,
    d_out_per_slice_cpu: torch.Tensor,
    slice_start_loc_cpu: torch.Tensor,
    w_lora_strides_cpu: torch.Tensor,
    add_inputs: bool,
) -> None:
    if no_lora_flag_cpu.item():
        return

    num_tokens = x.size(1)
    num_slices = len(lora_b_weights)
    total_d_out = y.size(1)

    # Prepare slice metadata
    for i, s in enumerate(output_slices):
        d_out_per_slice_cpu[i] = s
    d_out_per_slice_buffer[:num_slices].copy_(d_out_per_slice_cpu[:num_slices],
                                              non_blocking=True)

    slice_start = offset_start
    for i in range(num_slices):
        slice_start_loc_cpu[i] = slice_start
        if i < num_slices - 1:
            slice_start += output_slices[i]
    slice_start_loc_buffer[:num_slices].copy_(slice_start_loc_cpu[:num_slices],
                                              non_blocking=True)

    for i, w in enumerate(lora_b_weights):
        w_lora_strides_cpu[i] = w.stride(0)
    w_lora_strides_buffer[:num_slices].copy_(w_lora_strides_cpu[:num_slices],
                                             non_blocking=True)

    y_sorted = y_sorted_buffer[:num_tokens, :total_d_out]

    torch.ops._lora_C.dispatch_sgmv_expand_vllm(
        y, x, lora_b_weights, lora_token_start_loc, lora_ids,
        d_out_per_slice_buffer[:num_slices],
        slice_start_loc_buffer[:num_slices],
        w_lora_strides_buffer[:num_slices], cutlass_tmp, token_indices_sorted,
        y_sorted, w_ptr_buffer, add_inputs)


def _cutlass_expand_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_slices: list[int],
    offset_start: int,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    y_sorted_buffer: torch.Tensor,
    d_out_per_slice_buffer: torch.Tensor,
    slice_start_loc_buffer: torch.Tensor,
    w_lora_strides_buffer: torch.Tensor,
    d_out_per_slice_cpu: torch.Tensor,
    slice_start_loc_cpu: torch.Tensor,
    w_lora_strides_cpu: torch.Tensor,
    add_inputs: bool,
) -> None:
    return


# Register custom ops for torch.compile compatibility
try:
    direct_register_custom_op(
        op_name="cutlass_shrink",
        op_func=_cutlass_shrink,
        mutates_args=["y"],
        fake_impl=_cutlass_shrink_fake,
    )
    cutlass_shrink = torch.ops.vllm.cutlass_shrink
except AttributeError:
    cutlass_shrink = _cutlass_shrink

try:
    direct_register_custom_op(
        op_name="cutlass_expand",
        op_func=_cutlass_expand,
        mutates_args=["y"],
        fake_impl=_cutlass_expand_fake,
    )
    cutlass_expand = torch.ops.vllm.cutlass_expand
except AttributeError:
    cutlass_expand = _cutlass_expand


@torch.inference_mode()
def _bgmv_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    indices: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
) -> None:
    """
    BGMV shrink kernel wrapper.
    
    Args:
        y: Output tensor [num_slices, num_tokens, lora_rank]
        x: Input tensor [num_tokens, hidden_size]
        lora_a_weights: List of weight tensors, each [num_loras, 1, lora_rank, hidden_size]
        indices: Per-token LoRA indices [num_tokens], int64
        no_lora_flag_cpu: CPU tensor indicating if no LoRA is active
        w_ptr_buffer: Pre-allocated buffer for weight pointers [num_slices], int64
    """
    if no_lora_flag_cpu.item():
        return

    num_slices = len(lora_a_weights)
    
    # Check if we're in cudagraph capture mode
    # During capture, skip CPU-to-GPU copies as they're not allowed
    # The buffers should already be populated from warmup
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    if not is_capturing:
        # Prepare weight pointers - copy to GPU buffer for cudagraph compatibility
        w_ptrs = torch.tensor(
            [w.data_ptr() for w in lora_a_weights],
            dtype=torch.int64,
            device='cpu'
        )
        w_ptr_buffer[:num_slices].copy_(w_ptrs, non_blocking=False)
    
    torch.ops._lora_C.dispatch_bgmv_shrink_sliced(
        y, x, w_ptr_buffer[:num_slices], indices
    )


def _bgmv_shrink_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    indices: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
) -> None:
    return


@torch.inference_mode()
def _bgmv_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_slices: list[int],
    offset_start: int,
    indices: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    d_out_per_slice_buffer: torch.Tensor,
    slice_start_loc_buffer: torch.Tensor,
    d_out_per_slice_cpu: torch.Tensor,
    slice_start_loc_cpu: torch.Tensor,
    add_inputs: bool,
) -> None:
    """
    BGMV expand kernel wrapper.
    
    Args:
        y: Output tensor [num_tokens, total_d_out]
        x: Input tensor [num_slices, num_tokens, lora_rank]
        lora_b_weights: List of weight tensors, each [num_loras, 1, d_out_slice, lora_rank]
        output_slices: Output dimension per slice
        offset_start: Starting column offset in output
        indices: Per-token LoRA indices [num_tokens], int64
        no_lora_flag_cpu: CPU tensor indicating if no LoRA is active
        w_ptr_buffer: Pre-allocated buffer for weight pointers [num_slices], int64
        d_out_per_slice_buffer: GPU buffer for output dims [num_slices], int32
        slice_start_loc_buffer: GPU buffer for slice offsets [num_slices], int64
        d_out_per_slice_cpu: CPU pinned buffer for output dims
        slice_start_loc_cpu: CPU pinned buffer for slice offsets
        add_inputs: Whether to add to existing y values
    """
    if no_lora_flag_cpu.item():
        return

    num_slices = len(lora_b_weights)
    
    # Check if we're in cudagraph capture mode
    # During capture, skip CPU-to-GPU copies as they're not allowed
    # The buffers should already be populated from warmup
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    if not is_capturing:
        # Prepare weight pointers - copy to GPU buffer
        # This happens during warmup before cudagraph capture
        w_ptrs = torch.tensor(
            [w.data_ptr() for w in lora_b_weights],
            dtype=torch.int64,
            device='cpu'
        )
        w_ptr_buffer[:num_slices].copy_(w_ptrs, non_blocking=False)
        
        # Prepare slice_start_loc buffer
        # This is populated during warmup, values are fixed for a given config
        slice_start = offset_start
        for i in range(num_slices):
            slice_start_loc_cpu[i] = slice_start
            slice_start += output_slices[i]
        slice_start_loc_buffer[:num_slices].copy_(slice_start_loc_cpu[:num_slices],
                                                  non_blocking=False)
    
    # Call C++ kernel with pre-populated GPU buffer
    torch.ops._lora_C.dispatch_bgmv_expand_sliced(
        y, x, w_ptr_buffer[:num_slices], indices,
        slice_start_loc_buffer[:num_slices], output_slices
    )
    
    # Note: BGMV kernel writes directly to y, no scatter needed
    # If add_inputs is False, caller should zero y before calling


def _bgmv_expand_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_slices: list[int],
    offset_start: int,
    indices: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    d_out_per_slice_buffer: torch.Tensor,
    slice_start_loc_buffer: torch.Tensor,
    d_out_per_slice_cpu: torch.Tensor,
    slice_start_loc_cpu: torch.Tensor,
    add_inputs: bool,
) -> None:
    return


# Register BGMV custom ops for torch.compile compatibility
try:
    direct_register_custom_op(
        op_name="bgmv_shrink",
        op_func=_bgmv_shrink,
        mutates_args=["y"],
        fake_impl=_bgmv_shrink_fake,
    )
    bgmv_shrink = torch.ops.vllm.bgmv_shrink
except AttributeError:
    bgmv_shrink = _bgmv_shrink

try:
    direct_register_custom_op(
        op_name="bgmv_expand",
        op_func=_bgmv_expand,
        mutates_args=["y"],
        fake_impl=_bgmv_expand_fake,
    )
    bgmv_expand = torch.ops.vllm.bgmv_expand
except AttributeError:
    bgmv_expand = _bgmv_expand