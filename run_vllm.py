# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams
os.environ["VLLM_USE_DEEP_GEMM"]="1"
os.environ["VLLM_LOGGING_LEVEL"]="DEBUG"
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",
    "50% Off + free shipping' offer is based on 50% off and free shipping on your first box, plus 20% off your next 4 boxes. '55% Off + free shipping' offer is based on 55% off and free shipping on your first box, plus 20% off your next 4 boxes. 'Up to $130 Off' is based on a total discount applied over a 6-week period for an 8-meal per week subscription. Discounts vary for other meal plans and sizes. Not valid on premiums, meal upgrades, add-ons, taxes or shipping fees. May not be combined with gift cards or any other promotion. No cash value. Void outside the U.S. and where prohibited. Offer cannot be sold or otherwise bartered. Factor has the right to end or modify any offer at any time. Additional restrictions may apply. See https://www.factor75.com/terms for more details.",

]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
if __name__ == "__main__":

    model_id = "/home/ubuntu/models/Llama-4-Maverick-17B-128E-Instruct-FP8"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=8,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
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