#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

struct ExpandKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_2, _1, _1>;
};

struct ShrinkKernelConfig {
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_2, _1, _1>;
};
