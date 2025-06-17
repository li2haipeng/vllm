export VLLM_PROFILE_START_STOP="30-35"

nsys profile \
  -t nvtx,cuda \
  --cudabacktrace=all \
  --cuda-graph-trace=node \
  --wait all \
  --capture-range cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  -o logs/nsys_$(date +%Y%m%d_%H%M%S).nsys-rep \
    python benchmarks/benchmark_throughput.py \
    --backend vllm \
    --async-engine \
    --model /home/ubuntu/models/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --num-prompts 1 \
    --input-len 1600 \
    --output-len 600 \
    --max-num-seqs 1 \
    --max_model_len 131072 \
    --gpu-memory-utilization 0.95 \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet.txt \
    -tp 8
# ncu --nvtx \
#     --target-processes all \
#     --set full \
#     --replay-mode app-range \
#     -f -o output.ncu-rep \
#   python benchmarks/benchmark_throughput.py \
#     --backend vllm \
#     --async-engine \
#     --model /home/ubuntu/models/Llama-4-Scout-17B-16E-Instruct-FP8-OS_routed \
#     --num-prompts 1 \
#     --input-len 1600 \
#     --output-len 600 \
#     --max-num-seqs 1 \
#     --max_model_len 131072 \
#     --gpu-memory-utilization 0.95 \
#     --dataset-name sonnet \
#     --dataset-path benchmarks/sonnet.txt \
#     -tp 8