vllm bench serve \
  --model /home/ubuntu/models/Qwen3-32B \
  --lora-modules adapter0 adapter1 adapter2 adapter3 adapter4 adapter5 adapter6 adapter7 adapter8 \
                adapter9 adapter10 adapter11 adapter12 adapter13 adapter14 adapter15 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --max-concurrency 8 \
  --num-prompts 80

# adapter1 adapter2 adapter3 adapter4 adapter5 adapter6 adapter7 adapter8 \
#                 adapter9 adapter10 adapter11 adapter12 adapter13 adapter14 adapter15 adapter16\