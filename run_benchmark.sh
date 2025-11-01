#!/bin/bash
# -----------------------------
# Configuration Section
# -----------------------------
#Output Directory
export VLLM_TUNED_CONFIG_FOLDER=/home/ubuntu/workplace/KernelTuner/src/KernelTuner/Triton/multi_lora/configs/qwen3-32B-new
RESULTS_BASE_DIR="/home/ubuntu/workplace/LlmPerf/src/LlmPerf/results/Qwen3-32B-eagle2-lora"
PORT=8000

MODEL_NAME=/home/ubuntu/models/Qwen3-32B
SPECULATIVE_MODEL_PATH=/home/ubuntu/models/eagles/qwen3-32b-fp8-eagle2
TOKENIZER_NAME=/home/ubuntu/models/Qwen3-32B
NUM_SPECULATIVE_TOKENS=(5) #(5 3 1) #(3 2 1) # (6 5 4 3 2 1 0)
TENSOR_PARALLEL_SIZES=(4)

# Conda Path
CONDA_INIT_SCRIPT="/home/ubuntu/anaconda3/etc/profile.d/conda.sh"

# LlmPerf Configs
LLMPERF_ENV="llmperf"
LLMPERF_BENCHMARKING_CODE_PATH="/home/ubuntu/workplace/LlmPerf/src/LlmPerf/token_benchmark_ray.py"
DATASET_BASE_PATH="/home/ubuntu/workplace/BISBenchmarking/src/BISBenchmarking/datasets"
DATASETS=(
  'HumanEval_python_sample_1000' 'gsm8k_sample_1000' 'mtbench_80_json' 
)
CONCURRENCY_VALUES=(1 8 16)
MEAN_OUTPUT_TOKENS=(800)
STDDEV_OUTPUT_TOKENS=150
TIMEOUT=3600 #21600
MAX_NUM_COMPLETED_REQUESTS=100 #1000
SAMPLING_PARAMS='{"temperature": 0, "seed": 11111, "adapter": "adapter"}'
SONNET_INPUT_TOKENS_VALUES=(1600)
SONNET_INPUT_TOKENS_STD=150
LM_HEAD_QUANTIZATION=("fp16")
# -----------------------------
# Function Definitions
# -----------------------------

LORA_PATH=/home/ubuntu/models/loras-prod/tuned-qwen3-32b

start_vllm_server() {
    source $CONDA_INIT_SCRIPT
    conda activate vllm
    num_speculative_tokens=$1
    tensor_parallel_size=$2
    WORK_DIR="$RESULTS_BASE_DIR/tp_${tensor_parallel_size}/speculative_tokens_$num_speculative_tokens"
    LOG_FILE="$WORK_DIR/vllm_server.log"
    mkdir -p $WORK_DIR
    
    # -------- LOGGING --------
    echo "[INFO] ====================================================="
    echo "Starting vLLM OpenAI API server with the following configuration:"
    echo "Num Speculative Tokens: $num_speculative_tokens"
    echo "vLLM Log File Path  : $LOG_FILE"

    ## vllm server command
    vllm_args=(
        --no-enable-prefix-caching \
        --tensor-parallel-size $tensor_parallel_size \
        --compilation-config '{"compile_sizes": [1,2,4,8,16,32,64]}' \
        --tool-call-parser hermes \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --port $PORT
        --enable-lora --max-loras 16 \
        --lora-modules \
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
        --max-lora-rank 32 \
        )

    if [ "$num_speculative_tokens" -gt 0 ]; then
        vllm_args+=(
            --speculative-config "{\"method\": \"eagle\", \"model\": \"$SPECULATIVE_MODEL_PATH\", \"num_speculative_tokens\": $num_speculative_tokens}"
        )
    fi

    vllm serve $MODEL_NAME "${vllm_args[@]}" &> "$LOG_FILE" &
    VLLM_PID=$!
    echo "[INFO] vLLM server PID: $VLLM_PID"
    
    # Wait until the server is ready
    wait_for_server "http://localhost:${PORT}"

}

kill_vllm() {
    echo "[INFO] Killing all vLLM servers..."
    kill $VLLM_PID
    wait 5
    echo "====================================================="
}

wait_for_server() {
  url=$1
  echo "[INFO] Waiting for server at $url..."
  # Wait for the server to respond with a 200 status code
  while true; do
    response=$(curl --write-out "%{http_code}" --silent --output /dev/null $url/v1/models)
    if [[ "$response" -eq 200 ]]; then
        echo "[INFO] vLLM server is ready."
        conda deactivate
        break
    else
        echo "[INFO] Not ready yet (status code: $response). Retrying in 10s..."
        sleep 10
    fi
  done
}

run_benchmarking_sonnet(){
    export MODEL_ENDPOINT="http://localhost:${PORT}/v1"
    export OPENAI_API_KEY='1234'

    num_speculative_tokens=$1
    sonnet_input_tokens=$2
    tensor_parallel_size=$3
    echo "---------------------------------------------"
    echo "Starting benchmark script..."

    echo "Activating conda environment: $LLMPERF_ENV"
    source $CONDA_INIT_SCRIPT
    conda activate $LLMPERF_ENV
    
    echo "Running benchmark with Sonnet input values: $sonnet_input_tokens"
    RESULTS_DATASET_DIR="$RESULTS_BASE_DIR/speculative_tokens_$num_speculative_tokens/sonnet_$sonnet_input_tokens/"
    
    for concurrency in "${CONCURRENCY_VALUES[@]}"; do
        echo "Running benchmark with $concurrency concurrent requests..."
        start_time=$(date +%s)
        RESULT_SUBDIR="$RESULTS_DATASET_DIR/tp_${tensor_parallel_size}/concurrency_$concurrency"
        LOG_FILE="$RESULT_SUBDIR/run_log.log"
        mkdir -p "$RESULT_SUBDIR"
        python "$LLMPERF_BENCHMARKING_CODE_PATH" \
            --model "$MODEL_NAME" \
            --mean-input-tokens "$sonnet_input_tokens" \
            --stddev-input-tokens "$SONNET_INPUT_TOKENS_STD" \
            --mean-output-tokens "$MEAN_OUTPUT_TOKENS" \
            --stddev-output-tokens "$STDDEV_OUTPUT_TOKENS" \
            --num-concurrent-requests "$concurrency" \
            --tokenizer "$TOKENIZER_NAME" \
            --timeout "$TIMEOUT" \
            --max-num-completed-requests "$MAX_NUM_COMPLETED_REQUESTS" \
            --results-dir "$RESULT_SUBDIR" \
            --additional-sampling-params "$SAMPLING_PARAMS" \
            --llm-api openai \
            --count-output-tokens-with-tokenizer \
            &> "$LOG_FILE"

        end_time=$(date +%s)
        total_time=$((end_time - start_time))
        total_minutes=$((total_time / 60))
        remaining_seconds=$((total_time % 60))   
        echo "Finished: Speculative Tokens=$num_speculative_tokens | Dataset=$dataset | Concurrency=$concurrency"
        echo "Results stored in: $RESULT_SUBDIR"
        echo " Total time taken: ${total_minutes} minutes and ${remaining_seconds} seconds" 
        echo "---------------------------------------------"
    done
    
    curl http://localhost:${PORT}/metrics > "$RESULTS_DATASET_DIR/metrics.log"
}

run_benchmarking_dataset(){
    export OPENAI_API_BASE="http://localhost:${PORT}/v1"
    export OPENAI_API_KEY='1234'

    num_speculative_tokens=$1
    dataset=$2
    tensor_parallel_size=$3

    echo "Activating conda environment: $LLMPERF_ENV"
    source $CONDA_INIT_SCRIPT
    conda activate $LLMPERF_ENV

    echo "---------------------------------------------"
    echo "Starting benchmark script..."

    echo "Running benchmark with Dataset: $dataset"
    dataset_path="$DATASET_BASE_PATH/$dataset.json"
    RESULTS_DATASET_DIR="$RESULTS_BASE_DIR/speculative_tokens_$num_speculative_tokens/$dataset"
    for concurrency in "${CONCURRENCY_VALUES[@]}"; do
        echo "Running benchmark with $concurrency concurrent requests..."
        start_time=$(date +%s)
        RESULT_SUBDIR="$RESULTS_DATASET_DIR/tp_${tensor_parallel_size}/concurrency_$concurrency"
        LOG_FILE="$RESULT_SUBDIR/run_log.log"
        mkdir -p "$RESULT_SUBDIR"
        python "$LLMPERF_BENCHMARKING_CODE_PATH" \
            --model "$MODEL_NAME" \
            --mean-output-tokens "$MEAN_OUTPUT_TOKENS" \
            --stddev-output-tokens "$STDDEV_OUTPUT_TOKENS" \
            --num-concurrent-requests "$concurrency" \
            --tokenizer "$TOKENIZER_NAME" \
            --timeout "$TIMEOUT" \
            --max-num-completed-requests "$MAX_NUM_COMPLETED_REQUESTS" \
            --dataset "$dataset_path" \
            --results-dir "$RESULT_SUBDIR" \
            --additional-sampling-params "$SAMPLING_PARAMS" \
            --llm-api openai \
            --count-output-tokens-with-tokenizer \
            &> "$LOG_FILE"

        end_time=$(date +%s)
        total_time=$((end_time - start_time))
        total_minutes=$((total_time / 60))
        remaining_seconds=$((total_time % 60))   
        echo "Finished: Speculative Tokens=$num_speculative_tokens | Dataset=$dataset | Concurrency=$concurrency"
        echo "Results stored in: $RESULT_SUBDIR"
        echo " Total time taken: ${total_minutes} minutes and ${remaining_seconds} seconds" 
        echo "---------------------------------------------"
    done
    
    curl http://localhost:${PORT}/metrics > "$RESULTS_DATASET_DIR/metrics.log"

    conda deactivate
    sleep 5
}

echo "[INFO] ====================================================="
echo "[INFO] LLMPerf Benchmark Configuration"
echo "  - Benchmark Script          : $LLMPERF_BENCHMARKING_CODE_PATH"
echo "  - Datasets                  : ${DATASETS[*]}"
echo "  - Sonnet input tokens       : ${SONNET_INPUT_TOKENS_VALUES[*]}"
echo "  - Concurrency Levels        : ${CONCURRENCY_VALUES[*]}"
echo "  - Tensor parallel sizes     : ${TENSOR_PARALLEL_SIZES[*]}"
echo "  - Mean Output Tokens        : $MEAN_OUTPUT_TOKENS"
echo "  - Stddev Output Tokens      : $STDDEV_OUTPUT_TOKENS"
echo "  - Timeout (s)               : $TIMEOUT"
echo "  - Max Completed Requests    : $MAX_NUM_COMPLETED_REQUESTS"
echo "  - Dataset Base Path         : $DATASET_BASE_PATH"
echo "  - Results Base Directory    : $RESULTS_BASE_DIR"
echo "  - Sampling Params           : $SAMPLING_PARAMS"
echo "[INFO] ====================================================="

## SONNET BENCHMARKS
for sonnet_input_tokens in "${SONNET_INPUT_TOKENS_VALUES[@]}"; do
    for tensor_parallel_size in "${TENSOR_PARALLEL_SIZES[@]}"; do
        for num_speculative_tokens in "${NUM_SPECULATIVE_TOKENS[@]}"; do
            start_vllm_server "$num_speculative_tokens" "$tensor_parallel_size"
            run_benchmarking_sonnet "$num_speculative_tokens" "$sonnet_input_tokens" "$tensor_parallel_size"
            kill_vllm
        done
    done
done

for dataset in "${DATASETS[@]}"; do
    for tensor_parallel_size in "${TENSOR_PARALLEL_SIZES[@]}"; do
        for num_speculative_tokens in "${NUM_SPECULATIVE_TOKENS[@]}"; do
            start_vllm_server "$num_speculative_tokens" "$tensor_parallel_size"
            run_benchmarking_dataset "$num_speculative_tokens" "$dataset" "$tensor_parallel_size"
            kill_vllm
        done
    done
done

echo "All benchmarks completed."