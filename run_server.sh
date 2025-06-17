export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_FLASH_ATTN_VERSION=3
export VLLM_MLA_DISABLE=1
export VLLM_USE_V1=1

python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --port 8088 \
    --max-model-len 32768 --trust-remote-code \
    --compilation_config='{"cudagraph_capture_sizes": [1,4,8], "compile_sizes": [1,4,8]}'