BS=1
PROMPT_LEN=1600
TENSORBORD_DIR=/home/ubuntu/vllm/vllm_profile_v0
MODEL_PATH=/home/ubuntu/models
export VLLM_USE_V1=0
# export VLLM_USE_DEEP_GEMM=1
for MODEL in Llama-4-Maverick-17B-128E-Instruct-FP8
do
    echo "Running prefill profiling for model: $MODEL"
    python prefill_profile.py \
    --model $MODEL_PATH/$MODEL --tensor-parallel-size 8 --max-model-len 32768 \
    --compilation_config='{"cudagraph_capture_sizes": [1,4,8], "compile_sizes": [1,4,8]}' \
    --tensorboard $TENSORBORD_DIR --batch-size $BS \
    --prompt-len $PROMPT_LEN --vocab-size 128000 \
    run_num_steps -n 5 \
    
    
    
    echo "Running decode profiling for model: $MODEL"
    python decode_profile.py \
    --model $MODEL_PATH/$MODEL --tensor-parallel-size 8 --max-model-len 32768 \
    --compilation_config='{"cudagraph_capture_sizes": [1,4,8], "compile_sizes": [1,4,8]}' \
    --tensorboard $TENSORBORD_DIR --batch-size $BS \
    --prompt-len $PROMPT_LEN --vocab-size 128000 \
    run_num_steps -n 5 \
    
done

# python detailed_profile.py --model /home/ubuntu/models/DSR1 --prompt-len 1024 \
#                 --batch-size 1 --vocab-size 128000 --max_model_len 16384 --max_num_batched_tokens 16384 \
#                 --trust-remote-code \
#                 --tensorboard /home/ubuntu/vllm/vllm_profile_han_eager/ \
#                 --tensor-parallel-size 8 run_num_steps --num-steps 6