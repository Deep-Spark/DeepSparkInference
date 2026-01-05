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
import argparse as _argparse
# ====== PATCH: 兼容旧版 argparse 不支持 'deprecated' ======
_original_add_argument = _argparse._ArgumentGroup.add_argument

def _patched_add_argument(self, *args, **kwargs):
    kwargs.pop('deprecated', None)
    return _original_add_argument(self, *args, **kwargs)

_argparse._ArgumentGroup.add_argument = _patched_add_argument
# =========================================================
import io
import time
import argparse
import dataclasses
import inspect
from PIL import Image
import base64

# ==================== PATCH argparse to ignore 'deprecated' ====================
def make_action_compat(action_class):
    original_init = action_class.__init__
    def patched_init(self, *args, **kwargs):
        kwargs.pop('deprecated', None)
        original_init(self, *args, **kwargs)
    action_class.__init__ = patched_init

actions_to_patch = [
    argparse._StoreAction,
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
]

if hasattr(argparse, 'BooleanOptionalAction'):
    actions_to_patch.append(argparse.BooleanOptionalAction)

for action_cls in actions_to_patch:
    make_action_compat(action_cls)
# ==============================================================================

# Add parent path for utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from vllm import LLM, EngineArgs, SamplingParams
try:
    from utils import sampling_add_cli_args
except ImportError:
    # Fallback: define minimal sampling CLI args if utils missing
    def sampling_add_cli_args(parser):
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top-p", type=float, default=1.0)
        parser.add_argument("--max-tokens", type=int, default=16)
        return parser

def main():
    parser = argparse.ArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser = sampling_add_cli_args(parser)
    args = parser.parse_args()

    # --- Build EngineArgs properly ---
    engine_args = EngineArgs.from_cli_args(args)
    # Use dataclasses.asdict instead of .to_dict()
    engine_params = dataclasses.asdict(engine_args)

    # --- Build SamplingParams safely ---
    sampling_signature = inspect.signature(SamplingParams)
    sampling_arg_names = set(sampling_signature.parameters.keys())
    sampling_params = {
        k: v for k, v in vars(args).items()
        if k in sampling_arg_names and v is not None
    }
    sampling_params_obj = SamplingParams(**sampling_params)

    # --- Prepare input ---
    prompt = "Describe this image in one sentence."
    image_path = "./vllm_public_assets/cherry_blossom.jpg"
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Image not found: {image_path}")
        print("Please ensure the image exists or update the path.")
        sys.exit(1)

    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ],
    }]

    # --- Run inference ---
    print("🚀 Initializing LLM...")
    llm = LLM(**engine_params)

    print("🧠 Generating response...")
    start_time = time.perf_counter()
    outputs = llm.chat(messages, sampling_params=sampling_params_obj)
    end_time = time.perf_counter()
    duration = end_time - start_time

    # --- Output results ---
    num_tokens = 0
    for o in outputs:
        text = o.outputs[0].text
        num_tokens += len(o.outputs[0].token_ids)
        print(f"✅ Output: {text}")

    num_reqs = len(outputs)
    qps = num_reqs / duration if duration > 0 else 0
    token_per_sec = num_tokens / duration if duration > 0 else 0

    print(f"\n📊 Summary:")
    print(f"   Requests: {num_reqs}")
    print(f"   QPS: {qps:.2f}")
    print(f"   Total tokens: {num_tokens}")
    print(f"   Token/s: {token_per_sec:.2f}")
    print(f"   Duration: {duration:.2f}s")

if __name__ == "__main__":
    main()
