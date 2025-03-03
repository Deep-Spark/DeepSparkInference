# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import dataclasses
import inspect
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset


from vllm import LLM, EngineArgs, SamplingParams
import sys
from pathlib import Path
import os
from utils import sampling_add_cli_args

# LLaVA-1.5
def run_llava(question,engine_params,modality):
    assert modality == "image"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(**engine_params)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question,engine_params,modality):
    assert modality == "image"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    llm = LLM(**engine_params)
    stop_token_ids = None
    return llm, prompt, stop_token_ids

# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(question,engine_params,modality):
    assert modality == "video"
    prompt = f"USER: <video>\n{question}\nASSISTANT:"
    llm = LLM(**engine_params)
    stop_token_ids = None
    return llm, prompt, stop_token_ids

# LLaVA-onevision
def run_llava_onevision(question,engine_params,modality):

    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    llm = LLM(**engine_params)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision
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
        image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
        img_question = "What is the content of this image?"

        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        # Input video and question
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')
    parser = EngineArgs.add_cli_args(parser)
    parser = sampling_add_cli_args(parser)
    args = parser.parse_args()
    engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
    sampling_args = [
        param.name
        for param in list(
            inspect.signature(SamplingParams).parameters.values()
        )
    ]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    sampling_params = {
        attr: getattr(args, attr) for attr in sampling_args if args.__contains__(attr)
    }
    
    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = model_example_map[args.model_type](question,engine_params,args.modality)
    sampling_params['stop_token_ids'] = stop_token_ids

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(**sampling_params)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        } for _ in range(args.num_prompts)]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)