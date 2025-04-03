export VLLM_USE_DEEPGEMM=1
export VLLM_WORKER_MULTIPROC_METHOD='spawn'
# export VLLM_ATTENTION_BACKEND='FLASH_ATTN'
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/models/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --port 8088 \
    --trust-remote-code