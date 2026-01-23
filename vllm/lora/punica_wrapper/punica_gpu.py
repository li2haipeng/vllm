# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        fused_moe_lora,
        lora_expand,
        lora_shrink,
    )

from vllm import _custom_ops as ops
import vllm._lora_C  # noqa: F401 - triggers torch op registration

from .punica_base import PunicaWrapperBase




# ============================================================================
# CUTLASS LoRA kernels registered as custom ops for torch.compile compatibility
# ============================================================================
# These functions are registered as custom ops so torch.compile treats them
# as opaque - it won't trace into them, which allows us to use data_ptr()
# for caching metadata tensors (similar to Triton's approach).
# ============================================================================

@torch.inference_mode()
def _cutlass_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    token_indices_sorted_int64: torch.Tensor,
    scale: float,
) -> None:
    """
    CUTLASS SGMV shrink kernel wrapper.
    
    This function is registered as a custom op so torch.compile won't trace into it.
    """
    num_tokens = x.size(0)
    
    # Copy int32 indices to pre-allocated int64 buffer (fixed address for cudagraph)
    token_indices_sorted_int64[:num_tokens].copy_(token_indices_sorted[:num_tokens])
    
    # Use index_select which produces a contiguous output
    # Note: This creates a new tensor, but it's necessary because the C++ kernel
    # requires contiguous input. The tensor address will be captured in cudagraph.
    x_sorted = torch.index_select(x, 0, token_indices_sorted_int64[:num_tokens])
    
    # Call the C++ kernel
    torch.ops._lora_C.dispatch_sgmv_shrink_vllm(
        y, x_sorted, lora_a_weights,
        lora_token_start_loc, lora_ids,
        cutlass_tmp, w_ptr_buffer)


def _cutlass_shrink_fake(
    y: torch.Tensor,
    x: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    token_lora_mapping: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    cutlass_tmp: torch.Tensor,
    w_ptr_buffer: torch.Tensor,
    token_indices_sorted_int64: torch.Tensor,
    scale: float,
) -> None:
    """Fake implementation for torch.compile tracing."""
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
    # Pre-allocated GPU metadata buffers (for cudagraph compatibility)
    d_out_per_slice_buffer: torch.Tensor,
    slice_start_loc_buffer: torch.Tensor,
    w_lora_strides_buffer: torch.Tensor,
    # Pre-allocated pinned CPU buffers (source for cudaMemcpyAsync)
    d_out_per_slice_cpu: torch.Tensor,
    slice_start_loc_cpu: torch.Tensor,
    w_lora_strides_cpu: torch.Tensor,
    add_inputs: bool,
) -> None:
    """
    CUTLASS SGMV expand kernel wrapper.
    
    This function is registered as a custom op so torch.compile won't trace into it.
    Metadata tensors are pre-allocated (both GPU and pinned CPU) for cudagraph compatibility.
    """
    num_tokens = x.size(1)
    num_slices = len(lora_b_weights)
    total_d_out = y.size(1)
    
    # Update pinned CPU buffers (these have fixed addresses)
    # Then copy to GPU buffers (also fixed addresses)
    # This pattern is cudagraph-safe because both source and destination have stable addresses
    
    # d_out_per_slice
    for i, s in enumerate(output_slices):
        d_out_per_slice_cpu[i] = s
    d_out_per_slice_buffer[:num_slices].copy_(d_out_per_slice_cpu[:num_slices], non_blocking=True)
    
    # slice_start_loc
    slice_start = offset_start
    for i in range(num_slices):
        slice_start_loc_cpu[i] = slice_start
        if i < num_slices - 1:
            slice_start += output_slices[i]
    slice_start_loc_buffer[:num_slices].copy_(slice_start_loc_cpu[:num_slices], non_blocking=True)
    
    # w_lora_strides
    for i, w in enumerate(lora_b_weights):
        w_lora_strides_cpu[i] = w.stride(0)
    w_lora_strides_buffer[:num_slices].copy_(w_lora_strides_cpu[:num_slices], non_blocking=True)
    
    # Get a view of y_sorted_buffer with exact dimensions
    y_sorted = y_sorted_buffer[:num_tokens, :total_d_out]
    
    # Call the C++ kernel
    torch.ops._lora_C.dispatch_sgmv_expand_vllm(
        y, x, lora_b_weights,
        lora_token_start_loc, lora_ids,
        d_out_per_slice_buffer[:num_slices],
        slice_start_loc_buffer[:num_slices],
        w_lora_strides_buffer[:num_slices],
        cutlass_tmp,
        token_indices_sorted,
        y_sorted,
        w_ptr_buffer,
        add_inputs)


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
    """Fake implementation for torch.compile tracing."""
    return


# Register custom ops
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


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)

        self.max_loras = kwargs["max_loras"]
        self.max_num_batched_tokens = max_num_batched_tokens

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )

        # When speculative decoding is enabled, max_num_samples is
        # max_batches * (num_speculative_decoding_tokens + 1).
        # This line can be optimized by replacing max_num_batched_tokens
        # to  max_batches * (num_speculative_decoding_tokens + 1).
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )
        
        # Pre-allocated buffers for CUTLASS kernels (cudagraph compatibility)
        # These are sized for worst-case and reused across calls
        # Max slices is typically 3 (for qkv) but we use a larger value for safety
        self._max_slices = 8
        num_lora_indices = self.max_loras + 1
        
        # Temporary buffer for CUTLASS grouped GEMM
        tmp_size = num_lora_indices * self._max_slices * 1024
        self._cutlass_tmp = torch.zeros(tmp_size, dtype=torch.uint8, device=device)
        
        # Pre-allocated buffer for weight pointers (for cudagraph compatibility)
        # This avoids allocating inside C++ which breaks cudagraph
        self._w_ptr_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        
        # Pre-allocated y_sorted buffer for expand kernel
        # Size: max_num_batched_tokens x max_hidden_size
        # We use a large max_hidden_size to cover typical model sizes
        # This buffer MUST have fixed address for cudagraph compatibility
        self._max_hidden_size = 32768  # Should cover most models (e.g., 8192 for Llama-70B)
        # We allocate buffers for both fp16 and bf16 to avoid reallocation during cudagraph
        self._y_sorted_buffer_fp16 = torch.empty(
            (max_num_batched_tokens, self._max_hidden_size),
            dtype=torch.float16,
            device=device
        )
        self._y_sorted_buffer_bf16 = torch.empty(
            (max_num_batched_tokens, self._max_hidden_size),
            dtype=torch.bfloat16,
            device=device
        )
        
        # Pre-allocated metadata buffers for CUTLASS expand kernel
        # These have fixed addresses for cudagraph compatibility
        self._d_out_per_slice_buffer = torch.zeros(self._max_slices, dtype=torch.int32, device=device)
        self._slice_start_loc_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        self._w_lora_strides_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        
        # Pinned CPU buffers for metadata (source for cudaMemcpyAsync)
        # These must have fixed addresses for cudagraph compatibility
        self._d_out_per_slice_cpu = torch.zeros(self._max_slices, dtype=torch.int32, pin_memory=True)
        self._slice_start_loc_cpu = torch.zeros(self._max_slices, dtype=torch.int64, pin_memory=True)
        self._w_lora_strides_cpu = torch.zeros(self._max_slices, dtype=torch.int64, pin_memory=True)
        
        # Pre-allocated int64 buffer for token_indices_sorted (for index_select)
        # index_select requires int64 indices, but LoRAKernelMeta uses int32
        # This buffer has a fixed address for cudagraph compatibility
        self._token_indices_sorted_int64 = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )

    def update_metadata(
        self,
        mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ):
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1]).contiguous()
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            scale,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        lora_expand(
            x,
            lora_b_stacked,
            y,
            *self.token_mapping_meta.meta_args(num_tokens),
            offset_start=offset_start,
            add_inputs=True,
        )

        y = y.view_as(y_org)

    def add_cutlass_shrink(
            self,
            y: torch.Tensor,
            x: torch.Tensor,
            lora_a_stacked: tuple[torch.Tensor, ...],
            scale: float,
            **kwargs,
        ):
        """
        Performs GEMM for multiple slices of lora_a using CUTLASS.
        
        Uses vLLM's pre-allocated metadata tensors for cudagraph compatibility.
        Output y is in SORTED order (by lora_id). The expand kernel expects
        input in this sorted order.
        
        Args:
            y (torch.Tensor): Output tensors [num_slices, num_tokens, d_out]
            x (torch.Tensor): Input tensor [num_tokens, d_in]
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation (not used in CUTLASS yet)
        """
        x = x.view(-1, x.shape[-1])
        num_tokens = x.size(0)
        
        # Get metadata from LoRAKernelMeta (pre-allocated for cudagraph compatibility)
        (token_lora_mapping, token_indices_sorted, num_tokens_per_lora,
         lora_token_start_loc, active_lora_ids, no_lora_flag) = \
            self.token_mapping_meta.meta_args(num_tokens)
        
        # Call the custom op (registered for torch.compile compatibility)
        cutlass_shrink(
            y, x, list(lora_a_stacked),
            token_lora_mapping, token_indices_sorted, num_tokens_per_lora,
            lora_token_start_loc, active_lora_ids, no_lora_flag,
            self._cutlass_tmp, self._w_ptr_buffer,
            self._token_indices_sorted_int64, scale)


    def add_cutlass_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_b using CUTLASS.
        
        Uses vLLM's pre-allocated metadata tensors for cudagraph compatibility.
        Input x is in sorted order (from shrink output).
        Output is scattered back to original token order.
        
        Args:
            y (torch.Tensor): Output tensor [num_tokens, total_d_out]
            x (torch.Tensor): Input tensors [num_slices, num_tokens, d_in] in sorted order
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            offset_start (int): Starting column offset
            add_inputs (bool): Whether to add to existing y values
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)
        total_d_out = y.size(1)

        # Get metadata from LoRAKernelMeta (pre-allocated for cudagraph compatibility)
        (token_lora_mapping, token_indices_sorted, num_tokens_per_lora,
         lora_token_start_loc, active_lora_ids, no_lora_flag) = \
            self.token_mapping_meta.meta_args(num_tokens)
        
        # Select the correct y_sorted buffer based on dtype
        if x.dtype == torch.float16:
            y_sorted_buffer = self._y_sorted_buffer_fp16
        else:
            y_sorted_buffer = self._y_sorted_buffer_bf16
        
        # Call the custom op (registered for torch.compile compatibility)
        # All buffers (GPU and pinned CPU) are pre-allocated with fixed addresses
        cutlass_expand(
            y, x, list(lora_b_stacked),
            list(output_slices), offset_start,
            token_lora_mapping, token_indices_sorted, num_tokens_per_lora,
            lora_token_start_loc, active_lora_ids, no_lora_flag,
            self._cutlass_tmp, self._w_ptr_buffer, y_sorted_buffer,
            self._d_out_per_slice_buffer, self._slice_start_loc_buffer,
            self._w_lora_strides_buffer,
            self._d_out_per_slice_cpu, self._slice_start_loc_cpu,
            self._w_lora_strides_cpu,
            add_inputs)

        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked,),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        r = lora_b_stacked[0].size(-1)
        
        # Use same dtype as input for CUTLASS kernel
        # Note: buffer is zeroed inside the shrink kernel
        buffer = torch.empty(
            (len(output_slices), x.size(0), r), dtype=x.dtype, device=x.device
        )
        self.add_cutlass_shrink(
            buffer,
            x,
            lora_a_stacked,
            scale,
            **kwargs,
        )
        self.add_cutlass_expand(
            y,
            buffer,
            lora_b_stacked,
            output_slices,
            add_inputs=True,
            **kwargs,
        )

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty((x.size(0), r), dtype=torch.float32, device=x.device)

        lora_shrink(
            x,
            [lora_a_stacked],
            buffer.unsqueeze(dim=0),
            *self.prompt_mapping_meta.meta_args(x.size(0)),
            scale,
        )

        lora_expand(
            buffer.unsqueeze(dim=0),
            [lora_b_stacked],
            y,
            *self.prompt_mapping_meta.meta_args(buffer.size(0)),
            add_inputs=True,
        )
        y = y.view_as(y_org)

    def moe_lora_align_block_size(
        self,
        topk_ids: torch.Tensor,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution.
        """
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
        sorted_ids = torch.empty(
            (max_loras * max_num_tokens_padded,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
        # Expert ids must be set default to -1 to prevent a blank block
        expert_ids = torch.empty(
            (max_loras * max_num_m_blocks,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        num_tokens_post_pad = torch.empty(
            (max_loras), dtype=torch.int32, device=topk_ids.device
        )

        (token_lora_mapping, _, _, _, lora_ids, _) = self.token_mapping_meta.meta_args(
            num_tokens
        )

        ops.moe_lora_align_block_size(
            topk_ids,
            token_lora_mapping,
            num_experts,
            block_size,
            max_loras,
            max_num_tokens_padded,
            max_num_m_blocks,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            adapter_enabled,
            lora_ids,
        )
        if expert_map is not None:
            expert_ids = expert_map[expert_ids]

        return sorted_ids, expert_ids, num_tokens_post_pad

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
    ):
        """
        Performs a fused forward computation for LoRA of Mixture-of-Experts (MoE) layer.
        """
        (_, _, _, _, lora_ids, _) = self.token_mapping_meta.meta_args(x.size(0))
        fused_moe_lora(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_lora_rank,
            top_k_num,
            lora_ids,
            adapter_enabled,
            shrink_config.get("BLOCK_SIZE_M", 64),
            shrink_config.get("BLOCK_SIZE_N", 64),
            shrink_config.get("BLOCK_SIZE_K", 32),
            shrink_config.get("GROUP_SIZE_M", 8),
            shrink_config.get("NUM_WARPS", 4),
            shrink_config.get("NUM_STAGES", 3),
            shrink_config.get("SPLIT_K", 1),
            expand_config.get("BLOCK_SIZE_M", 64),
            expand_config.get("BLOCK_SIZE_N", 64),
            expand_config.get("BLOCK_SIZE_K", 32),
            expand_config.get("GROUP_SIZE_M", 8),
            expand_config.get("NUM_WARPS", 4),
            expand_config.get("NUM_STAGES", 3),
            expand_config.get("SPLIT_K", 1),
            mul_routed_weight,
            fully_sharded,
            offset,
        )
