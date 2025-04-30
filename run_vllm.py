# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile_simple"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
if __name__ == "__main__":

    # model_id = "/home/ubuntu/models/Llama-4-Maverick-17B-128E-Instruct-FP8/"
    # model_id = "nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8-quantized"
    # model_id = "nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8-Dynamic"
    # model_id = "/home/ubuntu/models/DSR1"
    # model_id = "/home/ubuntu/models/Llama-3.3-70B-Instruct-quantized.w4a16"
    # model_id = "/home/ubuntu/models/Llama-4-Scout-per-channel-w4a16"
    # model_id = "/home/ubuntu/models/Llama-4-Scout-per-group-w4a16"
    model_id = "/home/ubuntu/models/DSR1-awq"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=8,
        max_model_len=32768,
        enforce_eager=True
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
