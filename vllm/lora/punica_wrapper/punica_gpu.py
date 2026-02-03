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
from vllm.lora.utils import get_captured_lora_counts
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import round_up

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        fused_moe_lora,
        lora_expand,
        lora_shrink,
    )

from vllm import _custom_ops as ops

from .punica_base import PunicaWrapperBase

USE_CUTLASS_LORA = False
USE_BGMV_LORA = True

if USE_CUTLASS_LORA:
    from vllm.lora.ops.cuda_ops import(
        CutlassLoRAKernelMeta,
        cutlass_shrink,
        cutlass_expand,
    )

if USE_BGMV_LORA:
    from vllm.lora.ops.cuda_ops import(
        bgmv_shrink,
        bgmv_expand,
    )


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

        self.lora_config = kwargs["lora_config"]
        self.max_loras = self.lora_config.max_loras

        # Compute captured LoRA counts for cudagraph specialization.
        captured_lora_counts = get_captured_lora_counts(
            self.max_loras, self.lora_config.specialize_active_lora
        )

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras,
            max_num_batched_tokens,
            device=device,
            captured_lora_counts=captured_lora_counts,
        )

        # When speculative decoding is enabled, max_num_samples is
        # max_batches * (num_speculative_decoding_tokens + 1).
        # This line can be optimized by replacing max_num_batched_tokens
        # to  max_batches * (num_speculative_decoding_tokens + 1).
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras,
            max_num_batched_tokens,
            device=device,
            captured_lora_counts=captured_lora_counts,
        )
        
        # CUTLASS kernel metadata
        if USE_CUTLASS_LORA:
            self._init_cutlass_buffers(max_num_batched_tokens, device)
        
        # BGMV kernel buffers
        if USE_BGMV_LORA:
            self._init_bgmv_buffers(max_num_batched_tokens, device)

    def _init_cutlass_buffers(self, max_num_batched_tokens: int, device: torch.device | str):
        """Initialize pre-allocated buffers for CUTLASS kernels."""
        from vllm.lora.ops.cuda_ops import CutlassLoRAKernelMeta
        
        self.cutlass_token_mapping_meta = CutlassLoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )
        
        self._max_slices = 3
        num_lora_indices = self.max_loras + 1
        
        tmp_size = num_lora_indices * self._max_slices * 1024
        self._cutlass_tmp = torch.zeros(tmp_size, dtype=torch.uint8, device=device)
        # Separate w_ptr buffers for shrink and expand to avoid race conditions
        self._w_ptr_buffer_shrink = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        self._w_ptr_buffer_expand = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        
        # Max hidden size needs to accommodate large models
        # gate_up_proj can have output size up to 57344 (14336 * 4) or larger
        self._max_hidden_size = 65536
        self._y_sorted_buffer_fp16 = torch.empty(
            (max_num_batched_tokens, self._max_hidden_size),
            dtype=torch.float16, device=device
        )
        self._y_sorted_buffer_bf16 = torch.empty(
            (max_num_batched_tokens, self._max_hidden_size),
            dtype=torch.bfloat16, device=device
        )
        
        self._d_out_per_slice_buffer = torch.zeros(self._max_slices, dtype=torch.int32, device=device)
        self._slice_start_loc_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        self._w_lora_strides_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        
        self._d_out_per_slice_cpu = torch.zeros(self._max_slices, dtype=torch.int32, pin_memory=True)
        self._slice_start_loc_cpu = torch.zeros(self._max_slices, dtype=torch.int64, pin_memory=True)
        self._w_lora_strides_cpu = torch.zeros(self._max_slices, dtype=torch.int64, pin_memory=True)

    def _init_bgmv_buffers(self, max_num_batched_tokens: int, device: torch.device | str):
        """Initialize pre-allocated buffers for BGMV kernels."""
        self._max_slices = 3
        
        # BGMV uses per-token indices (int64) instead of sorted token groups
        self._bgmv_indices = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self._bgmv_w_ptr_buffer_shrink = torch.zeros(
            self._max_slices, dtype=torch.int64, device=device
        )
        self._bgmv_w_ptr_buffer_expand = torch.zeros(
            self._max_slices, dtype=torch.int64, device=device
        )
        # CPU tensor for no_lora flag (for cudagraph compatibility)
        self._bgmv_no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool, device="cpu")
        
        # Buffers for expand slice metadata
        self._bgmv_d_out_per_slice_buffer = torch.zeros(self._max_slices, dtype=torch.int32, device=device)
        self._bgmv_slice_start_loc_buffer = torch.zeros(self._max_slices, dtype=torch.int64, device=device)
        self._bgmv_d_out_per_slice_cpu = torch.zeros(self._max_slices, dtype=torch.int32, pin_memory=True)
        self._bgmv_slice_start_loc_cpu = torch.zeros(self._max_slices, dtype=torch.int64, pin_memory=True)

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
        
        # Prepare CUTLASS kernel metadata
        if USE_CUTLASS_LORA:
            self.cutlass_token_mapping_meta.prepare_tensors(self.token_lora_indices)
        
        # Prepare BGMV kernel metadata
        if USE_BGMV_LORA:
            # Prepare BGMV indices (int64 version of token_lora_indices)
            num_tokens = self.token_lora_indices.size(0)
            self._bgmv_indices[:num_tokens].copy_(
                self.token_lora_indices.to(torch.int64), non_blocking=True
            )
            # Update no_lora flag for BGMV
            self._bgmv_no_lora_flag_cpu[0] = torch.all(self.token_lora_indices == -1)

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
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
            *self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            ),
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

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """
        x = x.view(-1, x.shape[-1])
        num_tokens = x.size(0)
        
        (token_lora_mapping, token_indices_sorted, token_indices_sorted_int64,
         num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag) = \
            self.cutlass_token_mapping_meta.meta_args(num_tokens)
        
        cutlass_shrink(
            y, x, list(lora_a_stacked),
            token_lora_mapping, token_indices_sorted, token_indices_sorted_int64,
            num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag,
            self._cutlass_tmp, self._w_ptr_buffer_shrink, scale)

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
        num_tokens = x.size(1)

        (token_lora_mapping, token_indices_sorted, _,
         num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag) = \
            self.cutlass_token_mapping_meta.meta_args(num_tokens)
        
        y_sorted_buffer = self._y_sorted_buffer_fp16 if x.dtype == torch.float16 else self._y_sorted_buffer_bf16
        
        # Zero the y_sorted buffer to ensure tokens without LoRA contribute zero
        # Only zero the portion we'll use to minimize overhead
        total_d_out = y.size(1)
        y_sorted_buffer[:num_tokens, :total_d_out].zero_()
        
        cutlass_expand(
            y, x, list(lora_b_stacked),
            list(output_slices), offset_start,
            token_lora_mapping, token_indices_sorted,
            num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag,
            self._cutlass_tmp, self._w_ptr_buffer_expand, y_sorted_buffer,
            self._d_out_per_slice_buffer, self._slice_start_loc_buffer,
            self._w_lora_strides_buffer,
            self._d_out_per_slice_cpu, self._slice_start_loc_cpu,
            self._w_lora_strides_cpu,
            add_inputs)

        y = y.view_as(y_org)
    
    def add_bgmv_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_a using BGMV kernel.
        
        BGMV uses per-token indices directly without sorting, which can be
        more efficient for certain workloads.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors [num_slices, num_tokens, lora_rank]
            x (torch.Tensor): Input tensor [num_tokens, hidden_size]
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
                Each tensor has shape [num_loras, 1, lora_rank, hidden_size]
            scale (float): Scaling factor for the operation (applied in kernel)
        """
        x = x.view(-1, x.shape[-1])
        num_tokens = x.size(0)
        
        bgmv_shrink(
            y, x, list(lora_a_stacked),
            self._bgmv_indices[:num_tokens],
            self._bgmv_no_lora_flag_cpu,
            self._bgmv_w_ptr_buffer_shrink,
        )

    def add_bgmv_expand(
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
        Performs GEMM for multiple slices of lora_b using BGMV kernel.
        
        BGMV uses per-token indices directly without sorting.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor [num_tokens, total_d_out]
            x (torch.Tensor): Input tensors [num_slices, num_tokens, lora_rank]
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
                Each tensor has shape [num_loras, 1, d_out_slice, lora_rank]
            output_slices (tuple[int, ...]): Every slice's size
            offset_start (int): Starting column offset in output
            add_inputs (bool): Whether to add to existing y values. Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)
        
        bgmv_expand(
            y, x, list(lora_b_stacked),
            list(output_slices), offset_start,
            self._bgmv_indices[:num_tokens],
            self._bgmv_no_lora_flag_cpu,
            self._bgmv_w_ptr_buffer_expand,
            self._bgmv_d_out_per_slice_buffer,
            self._bgmv_slice_start_loc_buffer,
            self._bgmv_d_out_per_slice_cpu,
            self._bgmv_slice_start_loc_cpu,
            add_inputs,
        )

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
            *self.token_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
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

        if USE_BGMV_LORA:
            # BGMV path: use same dtype as input
            # Zero the buffer to ensure tokens without LoRA contribute zero
            buffer = torch.zeros(
                (len(output_slices), x.size(0), r), dtype=x.dtype, device=x.device
            )
            self.add_bgmv_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
            self.add_bgmv_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)
        elif USE_CUTLASS_LORA:
            # CUTLASS path: use same dtype as input
            # Zero the buffer to ensure tokens without LoRA contribute zero
            buffer = torch.zeros(
                (len(output_slices), x.size(0), r), dtype=x.dtype, device=x.device
            )
            self.add_cutlass_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
            self.add_cutlass_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)
        else:
            # Triton path: use float32 buffer
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            # Note: buffer is zeroed inside the shrink op
            buffer = torch.empty(
                (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
            )
            self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
            self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)

    # def add_lora_linear_bgmv(
    #     self,
    #     y: torch.Tensor,
    #     x: torch.Tensor,
    #     lora_a_stacked: tuple[torch.Tensor, ...],
    #     lora_b_stacked: tuple[torch.Tensor, ...],
    #     scale: float,
    #     output_slices: tuple[int, ...],
    #     *,
    #     buffer: torch.Tensor | None = None,
    #     **kwargs,
    # ) -> None:
    #     """
    #     Applicable to linear-related lora using BGMV kernels.
        
    #     BGMV kernels use per-token indices directly without sorting,
    #     which can be more efficient for certain workloads.

    #     Semantics:
    #         for i in range(len(lora_a_stacked)):
    #             y[i] += (
    #                 x[i].unsqueeze(0)
    #                 @ lora_a_stacked[indices[i], layer_idx, :, :]
    #                 @ lora_b_stacked[indices[i], layer_idx, :, :]
    #                 * scale
    #                 ).squeeze(0)
    #     Args:
    #         y (torch.Tensor): Output tensor. Will be changed in-place.
    #         x (torch.Tensor): Input tensor
    #         lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
    #         lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
    #         scale (float): Scaling factor.
    #         output_slices (tuple[int, ...]): Every slice's size.
    #         buffer (Optional[torch.Tensor]): Defaults to None.
    #     """
    #     assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

    #     assert buffer is None, (
    #         "To minimize overhead, the buffer should be created by "
    #         ".add_lora_linear_bgmv() instead of being passed in."
    #     )
    #     r = lora_b_stacked[0].size(-1)

    #     # BGMV path: use same dtype as input
    #     # Zero the buffer to ensure tokens without LoRA contribute zero
    #     buffer = torch.zeros(
    #         (len(output_slices), x.size(0), r), dtype=x.dtype, device=x.device
    #     )
    #     self.add_bgmv_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
    #     self.add_bgmv_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)

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
            *self.prompt_mapping_meta.meta_args(
                x.size(0), self.lora_config.specialize_active_lora
            ),
            scale,
        )

        lora_expand(
            buffer.unsqueeze(dim=0),
            [lora_b_stacked],
            y,
            *self.prompt_mapping_meta.meta_args(
                buffer.size(0), self.lora_config.specialize_active_lora
            ),
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
        naive_block_assignment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution.
        """
        (token_lora_mapping, _, _, _, lora_ids, _, _) = (
            self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            )
        )
        if naive_block_assignment:
            expert_ids = topk_ids.reshape(-1)
            sorted_ids = None
            num_tokens_post_pad = None
        else:
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

        return None, sorted_ids, expert_ids, num_tokens_post_pad

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ):
        """
        Performs a fused forward computation for LoRA of Mixture-of-Experts (MoE) layer.
        """
        (
            token_lora_mapping_meta,
            _,
            _,
            _,
            lora_ids,
            _,
            num_active_loras,
        ) = self.token_mapping_meta.meta_args(
            x.size(0), self.lora_config.specialize_active_lora
        )
        if token_lora_mapping is None:
            token_lora_mapping = token_lora_mapping_meta
        fused_moe_lora(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            max_lora_rank,
            top_k_num,
            lora_ids,
            num_active_loras,
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
