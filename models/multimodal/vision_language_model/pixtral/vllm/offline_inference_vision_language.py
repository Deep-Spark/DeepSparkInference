#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import sys
from pathlib import Path
import io
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import dataclasses
import inspect
from PIL import Image
import base64
from vllm import LLM, EngineArgs, SamplingParams

from utils import sampling_add_cli_args

# Pixtral
def run_pixtral(question,engine_params):

    prompt = prompt = f"{question}"
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(**engine_params)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    
    prompt = "Describe this image in one sentence."

    llm, prompt, stop_token_ids = run_pixtral(prompt,engine_params)
    sampling_params['stop_token_ids'] = stop_token_ids

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(**sampling_params)
    
    image: Image = Image.open("./vllm_public_assets/cherry_blossom.jpg")
    image = image.convert("RGB")
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_base64 = image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")

    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }
            ],
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)