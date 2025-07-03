# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os
import random
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

# Idefics3-8B-Llama3
def run_idefics3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "image"
    model_name = "./idefics3"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        enforce_eager=True,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
        mm_processor_kwargs={
            "size": {
                "longest_edge": 3 * 364
            },
        },
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )
    prompts = [(
        f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    ) for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

model_example_map = {
    "idefics3": run_idefics3,
}

def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = ImageAsset("cherry_blossom") \
            .pil_image.convert("RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_questions = ["Why is this video funny?"]

        return {
            "data": video,
            "questions": vid_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(image_repeat_prob, num_prompts, data,
                       prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob". 
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert (image_repeat_prob <= 1.0 and image_repeat_prob >= 0)
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({
            "prompt": prompts[i % len(prompts)],
            "multi_modal_data": {
                modality: cur_image
            }
        })

    return inputs


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality)

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)

    # To maintain code compatibility in this script, we add LoRA here.
    # You can also add LoRA using:
    # llm.generate(prompts, lora_request=lora_request,...)
    if req_data.lora_requests:
        for lora_request in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora_request)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = req_data.prompts if args.use_different_prompt_per_request else [
        req_data.prompts[0]
    ]

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=req_data.stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {
                modality: data
            },
        }
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs = apply_image_repeat(args.image_repeat_prob,
                                        args.num_prompts, data, prompts,
                                        modality)
        else:
            # Use the same image for all prompts
            inputs = [{
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {
                    modality: data
                },
            } for i in range(args.num_prompts)]

    if args.time_generate:
        import time
        start_time = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        print("-- generate time = {}".format(elapsed_time))

    else:
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=4,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Set the seed when initializing `vllm.LLM`.")

    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=None,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--disable-mm-preprocessor-cache',
        action='store_true',
        help='If True, disables caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')

    parser.add_argument(
        '--use-different-prompt-per-request',
        action='store_true',
        help='If True, then use different prompt (with the same multi-modal '
        'data) for each request.')

    args = parser.parse_args()
    main(args)