export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_FLASH_ATTN_VERSION=3
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1

python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Llama-4-Scout-per-group-w4a16 \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --port 8088 \
    --max-model-len 32768 --trust-remote-code \
    # --quantization moe_wna16
# vllm serve /home/ubuntu/models/DSR1 \
#       --served-model-name deepseek-ai/DeepSeek-R1 \
#       --max-model-len 16384 \
#       --max-seq-len-to-capture 16384 \
#       --max-num-seqs 8 \
#       --trust-remote-code \
#       --tensor-parallel-size 8  \
#       --port 8080 \
#       --max-num-batched-tokens 32768 \