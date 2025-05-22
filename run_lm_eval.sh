export model=/home/ubuntu/models/Llama-4-Scout-17B-16E-Instruct-FP8-OS_routed
lm_eval --model vllm \
        --model_args "pretrained=$model,tensor_parallel_size=8,max_model_len=100000" \
        --tasks  gsm8k \
        --batch_size auto \
        --seed 42 \
        --trust_remote_code \
        --limit 100