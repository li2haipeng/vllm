from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="/home/ubuntu/models/gpt-oss-120b", 
          enable_lora=True, 
          tensor_parallel_size=1, 
          enforce_eager=True,
          max_num_batched_tokens=16384,
          max_model_len=16384,
          max_lora_rank=32)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    # stop=["[/assistant]"]
)

prompts = [
    "Hello, my name is "
]
lora_path = "/home/ubuntu/models/loras/gpt-oss-120b-Lora/lora_adapter"
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 3, lora_path)
)
print(outputs[0].outputs[0].text)