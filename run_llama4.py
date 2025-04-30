import argparse
import os
import time
from vllm import LLM, SamplingParams
import torch
# os.environ["VLLM_USE_DEEP_GEMM"] = "1"
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile/images"
# os.environ["VLLM_ATTENTION_BACKEND"]="FLASH_ATTN"
# os.environ["VLLM_FLASH_ATTN_VERSION"]="3"
# os.environ["VLLM_MLA_DISABLE"] = "1"
# os.environ["VLLM_USE_V1"] = "0"
def test_text(llm):
    # Sample conversations.
    conversations = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        # [
        #     {"role": "system", "content": "Always answer with Haiku"},
        #     {"role": "user", "content": "I am going to Paris, what should I see?"},
        # ],
        # [
        #     {"role": "system", "content": "Always answer with emojis"},
        #     {"role": "user", "content": "How to go from Beijing to NY?"},
        # ],
        # [
        #     {"role": "user", "content": "I am going to Paris, what should I see?"},
        #     {
        #         "role": "assistant",
        #         "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.",
        #     },
        #     {"role": "user", "content": "What is so great about #1?"},
        # ],
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=4096)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # llm.start_profile()
    outputs = llm.chat(conversations, sampling_params)
    # llm.stop_profile()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")


def test_images(llm):
    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=4096)
    # Perform multi-image inference using llm.chat()
    # llm.start_profile()
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you describe how these two images are similar, and how they differ?",
                    },
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                        for image_url in image_urls
                    ),
                ],
            }
        ],
        sampling_params=sampling_params,
    )
    # llm.stop_profile()
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-----------------------------------")
        print(f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n")


def test_completion(llm):
    # Define sample prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling parameters object
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Generate texts from the prompts
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: (prompt!r), Generated text: {generated_text!r}")


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Run inference with a specified model ID."
    )

    # Add an argument for the model ID
    parser.add_argument(
        "model_id",
        type=str,
        help="The Hugging Face model ID to use (e.g., 'll-re/Llama-4-Scout-17B-16E-Instruct').",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the model_id argument
    model_id = args.model_id

    # Print or use the model_id as needed
    print(f"Using model: {model_id}")
    llm = LLM(
        model=model_id,
        enforce_eager=False,
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": 5},
        # Set max_model_len to 32768 just for testing
        max_model_len=32768,
    )
    if "instruct" in model_id.lower():
        print("---------Now start Instruct test-----------")
        
        # llm.start_profile()
        # torch.cuda.cudart().cudaProfilerStart()
        # torch.cuda.nvtx.range_push("range marker1")
        # test_images(llm)
        test_text(llm)
        # torch.cuda.nvtx.range_pop()
        # # torch.cuda.synchronize() if async
        # torch.cuda.cudart().cudaProfilerStop()
        # # llm.stop_profile()
    else:
        print("---------Now start Completion test-----------")
        test_completion(llm)


if __name__ == "__main__":
    main()
    # time.sleep(10)