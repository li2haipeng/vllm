#pragma once

// BGMV Kernel Configuration
// This file contains tunable parameters for shrink and expand kernels.
// Modify these values for kernel tuning without changing bgmv_impl.cuh

// Shrink kernel configuration - for shrink operations (h_in > h_out)
// Used when reducing from hidden_size to lora_rank
struct ShrinkKernelConfig {
    static constexpr int tx = 32;       // Threads in X (for warp shuffle), power of 2, <= 32
    static constexpr int ty = 4;        // Threads in Y (parallelism)
    static constexpr int vec_size = 8;  // Vector load size (4 or 8)
};

// Expand kernel configuration - for expand operations (h_in < h_out)
// Used when expanding from lora_rank to hidden_size
// Note: tx is derived from feat_in / vec_size at compile time
struct ExpandKernelConfig {
    static constexpr int tz = 4;        // Threads in Z (output parallelism)
    static constexpr int vec_size = 8;  // Vector load size (4 or 8)
};
