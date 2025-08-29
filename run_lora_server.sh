export LORA_PATH=/home/ubuntu/models/loras/qsq
vllm serve /home/ubuntu/models/Llama-3.3-70B-Instruct-FP8-dynamic-mlp-only \
    --trust-remote-code \
    -tp 8 \
    --enable-lora \
    --max-loras 16 \
    --lora-modules  \
        adapter0=$LORA_PATH \
        adapter1=$LORA_PATH \
        adapter2=$LORA_PATH \
        adapter3=$LORA_PATH \
        adapter4=$LORA_PATH \
        adapter5=$LORA_PATH \
        adapter6=$LORA_PATH \
        adapter7=$LORA_PATH \
        adapter8=$LORA_PATH \
        adapter9=$LORA_PATH \
        adapter10=$LORA_PATH \
        adapter11=$LORA_PATH \
        adapter12=$LORA_PATH \
        adapter13=$LORA_PATH \
        adapter14=$LORA_PATH \
        adapter15=$LORA_PATH \
    --max-lora-rank 32

# export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
# export VLLM_FLASH_ATTN_VERSION=3
# export VLLM_MLA_DISABLE=1
# export VLLM_USE_V1=1

# python -m vllm.entrypoints.openai.api_server \
#     --model /home/ubuntu/models/Llama-3.3-70B-Instruct-FP8-dynamic-mlp-only \
#     --tensor-parallel-size 8 \
#     --max-num-seqs 16 \
#     --port 8000 \
#     --max-model-len 32768 --trust-remote-code \
#     --compilation_config='{"cudagraph_capture_sizes": [1,4,8], "compile_sizes": [1,4,8]}'