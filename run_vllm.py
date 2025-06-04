# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams

# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
if __name__ == "__main__":

    model_id = "/home/ubuntu/models/Llama-4-Scout-17B-16E-Instruct-FP8-OS_routed"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=8,
        max_model_len=32768,
        # enforce_eager=True,
        # compilation_config={"cudagraph_capture_sizes": [1,4,8], "compile_sizes": [1,4,8]}
    )

    # llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    # llm.stop_profile()

    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)