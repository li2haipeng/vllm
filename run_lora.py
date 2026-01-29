from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="/home/ubuntu/models/Llama-3.3-70B-Instruct-FP8-dynamic-mlp-only", 
          enable_lora=True, 
          tensor_parallel_size=8, 
          enforce_eager=False,
          max_num_batched_tokens=16384,
          max_model_len=16384,
          max_lora_rank=32)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    # stop=["[/assistant]"]
)

prompts = [
    "Hello, my name is Lora, and I am a",
]
lora_path = "/home/ubuntu/models/loras/Llama-3.3-70B-Instruct-Lora/lora_adapter"
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, lora_path)
)
print(outputs[0].outputs[0].text)