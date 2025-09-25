# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for Punica kernel construction.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def mm_k(a_ptr, b_ptr, ak_stride, bk_stride, offset_k, K: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
         EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr, CAST_TYPE: tl.constexpr,
         b_dtype: tl.constexpr):
    """
    Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
    B (k x n), iterate, through the K dimension to compute the partial/complete
    matrix block product.
    If SPLIT_K == 1, the output m x n product is complete.
    If SPLIT_K > 1, the thread block computes partial outputs. The partial
    outputs are then atomically summed in the caller code. 
    Args:
        a_ptr: Array of pointers, identifying rows of A 
        b_ptr: Array of pointers, identifying columns of B
        ak_stride: K dimension stride of the A matrix
        bk_stride: K dimension stride of the B matrix
        K: Length of the K dimension
        BLOCK_M: M dimension of the output block m x n
        BLOCK_N: N dimension of the output block m x n
        BLOCK_K: K dimension atom
        EVEN_K: True if the blocks of A and B can be loaded without any
          masking.
        SPLIT_K: Parameter signifying parallelism in the K dimension. 
        CAST_TYPE: if True, cast the values from the A matrix to the B
          matrix dtype.
        b_dtype: datatype of the B matrix
    """
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :]
                              < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None]
                              < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(b_dtype)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * SPLIT_K * ak_stride
        b_ptr += BLOCK_K * SPLIT_K * bk_stride
    return accumulator


@triton.jit
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,  # array identifying the rows of Input ptr to operate on
    slice_start_loc,
    # input ptr strides
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    # lora ptr strides
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    # out ptr strides
    output_d0_stride,
    output_d1_stride,
    # constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    """
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice,
    compute the matrix product and store in the appropriate output location.
    Given that this is an expand kernel, we don't perform any split-K reduction
    as the K dimension is assumed to be small.
    """

    # ls_d*_ptr can be either an integer or a pointer
    if SAME_STRIDE:
        # integer
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        # pointer
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    # Identify the input_ptr and lora_ptr from slice_id.
    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty))

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (cur_input_ptr + ram[:, None] * input_d1_stride +
             offset_k[None, :] * input_d2_stride)
    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index +
             offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)

    # Compute the block matrix product.
    SPLIT_K = 1
    accumulator = mm_k(a_ptr, b_ptr, input_d2_stride, cur_lora_d2_stride,
                       offset_k, K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
                       CAST_TYPE, cur_lora_ptr.dtype.element_ty)

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (out_ptr + ram[:, None] * output_d0_stride +
             offset_cn[None, :] * output_d1_stride)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :]
                                             < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    # input strides
    input_d0_stride,
    input_d1_stride,
    # lora strides
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    # output strides
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
):
    """
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice, compute the
    matrix product and store in the appropriate output location.
    """

    # Identify the lora_ptr from slice_id.
    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (input_ptr + ram[:, None] * input_d0_stride +
             offset_k[None, :] * input_d1_stride)
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index +
             rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    # Compute partial/complete block matrix product.
    accumulator = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k,
                       K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, False,
                       cur_lora_ptr.dtype.element_ty)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr +
                   slice_id * output_d0_stride)
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[
        None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    accumulator *= scaling
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@triton.jit
def do_shrink_expand_kernel(
    pid_n,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_R: tl.constexpr,
    ADD_INPUTS: tl.constexpr
):
    """
    Fused shrink-expand kernel: computes Y = X @ A @ B where
    A is the shrink matrix (K x r) and B is the expand matrix (r x N)
    """
    if SAME_STRIDE:
        # integer
        cur_lora_b_d0_stride = ls_b_d0_ptr
        cur_lora_b_d1_stride = ls_b_d1_ptr
        cur_lora_b_d2_stride = ls_b_d2_ptr
    else:
        # pointer
        cur_lora_b_d0_stride = tl.load(ls_b_d0_ptr + slice_id)
        cur_lora_b_d1_stride = tl.load(ls_b_d1_ptr + slice_id)
        cur_lora_b_d2_stride = tl.load(ls_b_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_lora_a_ptr = lora_a_ptr
        cur_lora_b_ptr = lora_b_ptr
        slice_offset = 0
    else:
        # Load the pointers for the current slice
        cur_lora_a_ptr = tl.load(lora_a_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))
        cur_lora_b_ptr = tl.load(lora_b_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))
        # Get slice offset for output indexing
        slice_offset = tl.load(slice_start_loc + slice_id)
    

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % curr_N, BLOCK_N), BLOCK_N)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    
    offset_m = tl.arange(0, BLOCK_M)
    #TODO(@haipenl): currently only support Rs are the same for all LoRAs
    # for r_start in range(0, R, BLOCK_R):
        # if r_start < R:
    offset_r = tl.arange(0, BLOCK_R)
    # r_mask = offset_r < R
    
    # Step 1: Compute U = X @ A (shrink operation)
    u_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float16)
    
    # Loop over K dimension
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for k_block_id in range(num_k_blocks):
        k_start = k_block_id * BLOCK_K
        offset_k = tl.arange(0, BLOCK_K) + k_start
        
        x_ptr = (input_ptr + ram[:, None] * input_d0_stride +
                    offset_k[None, :] * input_d1_stride)
        
        a_ptr = (cur_lora_a_ptr + lora_id * ls_a_d0_ptr +
                    offset_k[:, None] * ls_a_d2_ptr +
                    offset_r[None, :] * ls_a_d1_ptr)
        if EVEN_K and EVEN_R:
            x_block = tl.load(x_ptr)
            a_block = tl.load(a_ptr)
        else:
            k_mask = offset_k < K
            x_mask = (offset_m[:, None] < cta_m_len) & k_mask[None, :]
            # a_mask = k_mask[:, None] & r_mask[None, :]
            
            x_block = tl.load(x_ptr, mask=x_mask, other=0.0)
            a_block = tl.load(a_ptr, mask=k_mask[:, None], other=0.0)
        
        u_acc += tl.dot(x_block, a_block, out_dtype=tl.float16)

    # u_acc = u_acc.to(tl.float16)
    
    # Step 2: Compute Y = U @ B (expand operation)
    b_ptr = (cur_lora_b_ptr + lora_id * cur_lora_b_d0_stride +
                offset_r[:, None] * cur_lora_b_d2_stride +
                rbn[None, :] * cur_lora_b_d1_stride)
    if EVEN_R:
        b_block = tl.load(b_ptr)
    else:
        # b_mask = r_mask[:, None] & (offset_n[None, :] < curr_N)  
        b_mask = offset_n[None, :] < curr_N
        b_block = tl.load(b_ptr, mask=b_mask, other=0.0)
    
    # u_masked = tl.where(r_mask[None, :], u_acc, 0.0)
    accumulator += tl.dot(u_acc, b_block, out_dtype=tl.float16)
    
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_offset
    c_ptr = output_ptr + ram[:, None] * output_d0_stride + offset_cn[None, :] * output_d1_stride
    c_mask = (offset_m[:, None] < cta_m_len) & (offset_cn[None, :] < (curr_N + slice_offset))
    
    if ADD_INPUTS:
        existing = tl.load(c_ptr, mask=c_mask, other=0.0)
        accumulator = accumulator + existing

    # # Apply final casting if needed
    # if CAST_TYPE:
    # accumulator = accumulator.to(tl.float16)
    
    # Store the result
    tl.store(c_ptr, accumulator, mask=c_mask)
