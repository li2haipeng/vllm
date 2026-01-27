#include "bgmv_config.h"
#include "bgmv_impl.cuh"

// Original BGMV instantiations
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_bfloat16)
FOR_INST_BGMV_WIDE_NARROW(INST_BGMV_ONESIDE, nv_bfloat16, nv_bfloat16, nv_bfloat16)

// Multi-slice BGMV instantiations (shrink: wide->narrow, expand: narrow->wide)
#define INST_BGMV_SLICED_SHRINK(in_T, out_T, W_T, narrow, wide) \
  INST_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T)

#define INST_BGMV_SLICED_EXPAND(in_T, out_T, W_T, narrow, wide) \
  INST_BGMV_EXPAND_SLICED(narrow, wide, in_T, out_T, W_T)

FOR_BGMV_WIDE_NARROW(INST_BGMV_SLICED_SHRINK, nv_bfloat16, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_SLICED_EXPAND, nv_bfloat16, nv_bfloat16, nv_bfloat16)
