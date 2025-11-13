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
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import dataclasses
import inspect
import logging
import time

import torch
from utils import load_chat_template, sampling_add_cli_args
from vllm import LLM, EngineArgs, SamplingParams

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

    model_name = os.path.dirname(args.model).rsplit("/")[-1]

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)

    # Create an LLM.
    llm = LLM(**engine_params)

    # skip process chat template
    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = llm.encode(prompts)
    # Print the outputs.
    for output in outputs:
        print(output.outputs.embedding) # list of hidden_size floats
        print("Offline inference is successful!")