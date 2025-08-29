# BS=1
# PROMPT_LEN=10000
# TENSORBORD_DIR=/home/ubuntu/vllm/lora_profile_test
# MODEL_PATH=/home/ubuntu/models
# export VLLM_DISABLE_BARRIERS=1

# for MODEL in Llama-3.3-70B-Instruct-FP8-dynamic-mlp-only
# do
#     echo "Running prefill profiling for model: $MODEL"
#     python profile_prefill.py \
#     --model $MODEL_PATH/$MODEL --tensor-parallel-size 8 \
#     --tensorboard $TENSORBORD_DIR --batch-size $BS \
#     --prompt-len $PROMPT_LEN --vocab-size 128000 \
#     --enable-lora \
#     --enforce-eager \
#     run_num_steps -n 5 \
    
    
    
#     echo "Running decode profiling for model: $MODEL"
#     python profile_decode.py \
#     --model $MODEL_PATH/$MODEL --tensor-parallel-size 8 \
#     --tensorboard $TENSORBORD_DIR --batch-size $BS \
#     --prompt-len $PROMPT_LEN --vocab-size 128000 \
#     --enable-lora \
#     --enforce-eager \
#     run_num_steps -n 5 \
    
# done
export VLLM_TORCH_PROFILER_DIR=/home/ubuntu/vllm/lora_profile_no_barrier
export VLLM_DISABLE_BARRIERS=1
python /home/ubuntu/vllm/benchmarks/benchmark_throughput.py \
            --model /home/ubuntu/models/Llama-3.3-70B-Instruct-FP8-dynamic-mlp-only \
            --tensor-parallel-size 8 \
            --trust-remote-code \
            --max-num-batched-tokens 262144 \
            --max-model-len 131072 \
            --max-seq-len-to-capture 131072 \
            --seed 43 \
            --no-enable-prefix-caching \
            --gpu-memory-utilization 0.95 \
            --num-prompts 10 \
            --max-num-seqs 8 \
            --input-len 1600 \
            --output-len 200 \
            # --output-json '/home/ubuntu/vllm/benchmarks/pytorch_profiling/benchmark_results.json'