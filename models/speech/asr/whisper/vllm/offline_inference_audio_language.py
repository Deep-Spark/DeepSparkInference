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
#
# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM for Whisper models using greedy sampling.

Run `pytest tests/models/encoder_decoder/audio/test_whisper.py`.
"""
import time
from typing import Optional

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import dataclasses
import inspect
from vllm import LLM, EngineArgs, SamplingParams
from utils import sampling_add_cli_args
from vllm.assets.audio import AudioAsset

PROMPTS = [
    {
        "prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    }
]

EXPECTED = {
    "openai/whisper-tiny": [
        " He has birth words I spoke in the original corner of that. And a"
        " little piece of black coat poetry. Mary had a little sandwich,"
        " sweet, with white and snow. And everyone had it very went the last"
        " would sure to go.",
        " >> And the old one, fit John the way to Edgar Martinez. >> One more"
        " to line down the field line for our base camp. Here comes joy. Here"
        " is June and the third base. They're going to wave him in. The throw"
        " to the plate will be late. The Mariners are going to play for the"
        " American League Championship. I don't believe it. It just continues"
        " by all five."
    ],
    "openai/whisper-small": [
        " The first words I spoke in the original pornograph. A little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite a"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the old one pitch on the way to Edgar Martinez one month. Here"
        " comes joy. Here is Junior to third base. They're gonna wave him"
        " in. The throw to the plate will be late. The Mariners are going to"
        " play for the American League Championship. I don't believe it. It"
        " just continues. My, oh my."
    ],
    "openai/whisper-medium": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite as"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez swung on the line"
        " down the left field line for Obeyshev. Here comes Joy. Here is"
        " Jorgen at third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh"
        " my."
    ],
    "openai/whisper-large-v3": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its feet were quite as"
        " slow, and everywhere that Mary went, the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line."
        " Now the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my."
    ],
    "openai/whisper-large-v3-turbo": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its streets were quite"
        " as slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line"
        " down the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my."
    ]
}


def run_whisper(engine_params,sampling_param,model_name) -> None:
    # import pdb
    # pdb.set_trace()
    prompt_list = PROMPTS * 10
    expected_list = EXPECTED[model_name] * 10

    llm = LLM(**engine_params)
    sampling_params = SamplingParams(**sampling_param)

    
    start_time = time.perf_counter()
    outputs = llm.generate(prompt_list, sampling_params)
    end_time = time.perf_counter()
    duration_time = end_time - start_time
    num_tokens = 0
    for output, expected in zip(outputs, expected_list):
        num_tokens += len(output.outputs[0].token_ids)
        print(output.outputs[0].text)
    num_requests = len(prompt_list)  # 请求的数量
    qps = num_requests / duration_time
    print(f"requests: {num_requests}, QPS: {qps}, tokens: {num_tokens}, Token/s: {num_tokens/duration_time}")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        help='Model name')
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
    run_whisper(engine_params,sampling_params, args.model_name)

