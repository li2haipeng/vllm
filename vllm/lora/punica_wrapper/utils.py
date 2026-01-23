# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping


def compute_meta(
    token_lora_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]:
    """
    Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function
    will combine them into a single request, improving sgmv kernel inference
    performance.
    2. At the beginning of each prefill stage inference, recalculations are
    needed based on the input, but only once.
    """

    lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
        token_lora_tensor, return_counts=True
    )
    cum_result = torch.cumsum(seq_length_tensor, dim=0)
    b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
    b_seq_start_tensor[1:].copy_(cum_result[:-1])
    max_length = seq_length_tensor.max().item()
    token_nums = seq_length_tensor.sum().item()
    batch_size = lora_indices_tensor.size(0)
    no_lora = False
    # -1 means no lora should be applied. Use `no_lora` to determine whether
    # the current step requires LoRA. If LoRA is not needed, the prefill stage
    # does not need to launch the triton kernel, which can improve performance
    if batch_size == 1 and lora_indices_tensor == -1:
        no_lora = True
    return (
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        token_nums,
        no_lora,
    )


# TODO see if this can be vectorized
def convert_mapping(
    mapping: "LoRAMapping",
    lora_index_to_id: list[int | None],
    max_loras: int,
    vocab_size: int,
    extra_vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indices. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indices, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            indices_len: List of lengths of the above tensors. It contains
                (base_indices, sampler_indices, sampler_indices_padded,
                embeddings_indices).
    """
    index_mapping_indices: list[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()

    prompt_mapping: list[int] = [
        lora_index_to_id.index(x) if x > 0 else -1 for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        # TODO index can be slow. optimize
        lora_idx = (
            lora_index_to_id.index(index_mapping_indices[i])
            if index_mapping_indices[i] > 0
            else -1
        )
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx

    indices_list: list[list[int] | torch.Tensor] = [
        index_mapping_indices,
        lora_indices,
        embedding_indices,
    ]

    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    prompt_mapping_tensor = torch.tensor(
        prompt_mapping, dtype=torch.long, device=device
    )
    embeddings_indices = torch.stack(
        [
            indices[2] * extra_vocab_size,
            indices[2] * (vocab_size + extra_vocab_size),
        ]
    )
    embeddings_indices = torch.where(
        embeddings_indices == -1, max_loras - 1, embeddings_indices
    )
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded = torch.where(
        sampler_indices_padded == -1, max_loras - 1, sampler_indices_padded
    )
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device=device, dtype=torch.long
    ) + (sampler_indices_padded * len(sampler_indices_padded))

    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    ]

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        indices_len,
    )

def convert_vllm_metadata_to_cutlass(
    token_lora_mapping: torch.Tensor,
    num_loras: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert vLLM-style token_lora_mapping to CUTLASS segment format.
    
    vLLM uses:
        - token_lora_mapping: [num_tokens] mapping each token to a lora_id
          (can contain -1 for tokens without LoRA)
    
    CUTLASS uses:
        - s: [num_loras + 1] segment boundaries (cumsum of tokens per lora)
        - Assumes tokens are already sorted by lora_id
    
    Args:
        token_lora_mapping: [num_tokens] tensor mapping tokens to lora_ids
        num_loras: number of active LoRA adapters
    
    Returns:
        token_indices_sorted_by_lora_ids: [num_tokens] reordered indices
        s: [num_loras + 1] segment boundaries for CUTLASS
        
    Note: Tokens with lora_id=-1 are sorted to the beginning but excluded
    from the segment boundaries. The kernel will only process tokens in
    segments for lora_ids 0 to num_loras-1.
    """
    device = token_lora_mapping.device
    
    # Sort token indices by lora_id (stable sort preserves order within same lora_id)
    # Tokens with -1 will be sorted to the beginning
    sorted_indices = torch.argsort(token_lora_mapping, stable=True)
    token_indices_sorted_by_lora_ids = sorted_indices.to(torch.int64)
    
    # Count tokens with no lora (lora_id == -1)
    num_no_lora_tokens = (token_lora_mapping == -1).sum().item()
    
    # Count tokens per lora (only for valid lora_ids 0 to num_loras-1)
    num_tokens_per_lora = torch.zeros(num_loras, dtype=torch.int64, device=device)
    for lora_id in range(num_loras):
        num_tokens_per_lora[lora_id] = (token_lora_mapping == lora_id).sum()
    
    # Compute start locations (cumsum), offset by num_no_lora_tokens
    # s[0] = num_no_lora_tokens (skip tokens with no lora)
    # s[i+1] = s[i] + num_tokens_per_lora[i]
    lora_token_start_loc = torch.zeros(num_loras + 1, dtype=torch.int64, device=device)
    lora_token_start_loc[0] = num_no_lora_tokens
    lora_token_start_loc[1:] = num_no_lora_tokens + torch.cumsum(num_tokens_per_lora, dim=0)
    
    # CUTLASS segment format: same as lora_token_start_loc but int32
    s = lora_token_start_loc.to(torch.int32)
    
    return (token_indices_sorted_by_lora_ids, s)


def reorder_input_by_lora(x: torch.Tensor, 
                          token_indices_sorted: torch.Tensor) -> torch.Tensor:
    """Reorder input tensor by sorted lora indices for CUTLASS."""
    return x[token_indices_sorted]

def scatter_output_by_lora(y_sorted: torch.Tensor,
                           token_indices_sorted: torch.Tensor,
                           original_shape: tuple) -> torch.Tensor:
    """Scatter CUTLASS output back to original token order."""
    if y_sorted.dim() == 2:
        y_original = torch.zeros(original_shape, dtype=y_sorted.dtype, device=y_sorted.device)
        y_original[token_indices_sorted] = y_sorted
    else:  # 3D: [num_slices, num_tokens, d_out]
        num_slices = y_sorted.size(0)
        y_original = torch.zeros(original_shape, dtype=y_sorted.dtype, device=y_sorted.device)
        for s in range(num_slices):
            y_original[s, token_indices_sorted] = y_sorted[s]
    return y_original