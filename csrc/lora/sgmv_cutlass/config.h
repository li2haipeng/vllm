// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

// Expand kernel: x @ W_B where x is [M, lora_rank], W_B is [lora_rank, hidden_size]
struct ExpandKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
};

// Shrink kernel: x @ W_A where x is [M, hidden_size], W_A is [hidden_size, lora_rank]
struct ShrinkKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
};
