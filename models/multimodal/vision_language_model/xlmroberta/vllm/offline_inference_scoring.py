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
import argparse as _argparse
# ====== PATCH: 兼容旧版 argparse 不支持 'deprecated' ======
_original_add_argument = _argparse._ArgumentGroup.add_argument

def _patched_add_argument(self, *args, **kwargs):
    kwargs.pop('deprecated', None)
    return _original_add_argument(self, *args, **kwargs)

_argparse._ArgumentGroup.add_argument = _patched_add_argument
# =========================================================

from vllm import LLM
import argparse
from vllm import LLM, EngineArgs
import dataclasses

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    engine_params = dataclasses.asdict(engine_args)
    # Sample prompts.
    text_1 = "What is the capital of France?"
    texts_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    # Create an LLM.
    # You should pass task="score" for cross-encoder models
    model = LLM(**engine_params)

    # Generate scores. The output is a list of ScoringRequestOutputs.
    outputs = model.score(text_1, texts_2)

    # Print the outputs.
    for text_2, output in zip(texts_2, outputs):
        score = output.outputs.score
        print(f"Pair: {[text_1, text_2]!r} | Score: {score}")
        
