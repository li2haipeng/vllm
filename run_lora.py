from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="/home/ubuntu/models/Qwen3-32B", 
          enable_lora=True, 
          max_loras=1,
          tensor_parallel_size=4, 
          enforce_eager=True,
          max_num_batched_tokens=16384,
          max_model_len=16384,
          max_lora_rank=32,
          speculative_config={
                "model": "/home/ubuntu/models/eagles/qwen3-32b-fp8-eagle2",
                "draft_tensor_parallel_size": 1,
                "num_speculative_tokens": 5,
                "method": "eagle",
            },
        )

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    # stop=["[/assistant]"]
)

prompts = [
    "Hello, my name is Lora, and I am a",
]
lora_path = "/home/ubuntu/models/loras/Qwen3-32B-Lora/lora_adapter"
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, lora_path)
)
print(outputs[0].outputs[0].text)