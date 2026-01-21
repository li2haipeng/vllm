#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

// Expand kernel configuration - for expand operations (typically larger N)
struct ExpandKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_2, _1, _1>;
};

// Shrink kernel configuration - for shrink operations (typically larger K)
struct ShrinkKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_2, _1, _1>;
};

