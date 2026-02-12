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

import sys
from pathlib import Path
import os
import argparse as _argparse
import dataclasses

# ====== PATCH: 兼容旧版 argparse 不支持 'deprecated' ======
_original_add_argument = _argparse._ArgumentGroup.add_argument

def _patched_add_argument(self, *args, **kwargs):
    kwargs.pop('deprecated', None)
    return _original_add_argument(self, *args, **kwargs)

_argparse._ArgumentGroup.add_argument = _patched_add_argument
# =========================================================

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import inspect
import logging
import time

import torch
from utils import load_chat_template, sampling_add_cli_args
from vllm import LLM, EngineArgs, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument(
        "--remove_chat_template",
        default=False,
        action="store_true",
        help="pass this if you are not use a chat model",
    )
    parser = EngineArgs.add_cli_args(parser)
    parser = sampling_add_cli_args(parser)
    args = parser.parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    engine_params = dataclasses.asdict(engine_args)  

    sampling_args = [
        param.name
        for param in inspect.signature(SamplingParams).parameters.values()
    ]
    sampling_params_dict = {
        attr: getattr(args, attr) for attr in sampling_args if hasattr(args, attr)
    }
    sampling_params = SamplingParams(**sampling_params_dict)

    model_name = os.path.dirname(args.model).rsplit("/")[-1]

    prompts = ["哪些迹象可能表明一个人正在经历焦虑?", "描述一下如何制作芝士披萨。", "写一篇有关5G网络研发的综述文章。"]

    # Create an LLM.
    llm = LLM(**engine_params)

    # Process chat template
    if args.remove_chat_template:
        if "chat" in model_name.lower():
            logging.warning(
                f"The model name from model path is {model_name}, so we guess you are using the chat model. "
                f"If the result is not quite correct, please do not pass --remove_chat_template."
            )
        prompts_new = prompts
    else:
        logging.warning("If you are using a non-chat model, please pass --remove_chat_template.")
        try:
            load_chat_template(llm.get_tokenizer(), args.chat_template)
            prompts_new = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = llm.get_tokenizer().apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts_new.append(text)
        except Exception as e:
            logging.warning(f"apply_chat_template failed: {e}. may because of low transformers version...(try use transformers>=4.34.0)")
            prompts_new = prompts

    # Warmup (optional but avoids first-run overhead in timing)
    _ = llm.generate(prompts_new[:1], sampling_params, use_tqdm=False)
    torch.cuda.synchronize()

    # Timed inference
    start_time = time.perf_counter()
    outputs = llm.generate(prompts_new, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time

    num_tokens = 0
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        num_tokens += len(output.outputs[0].token_ids)
        print(f"Prompt: {prompt}\nGenerated text: {generated_text}\n")

    num_requests = len(prompts_new)
    qps = num_requests / duration_time if duration_time > 0 else float('inf')
    token_per_sec = num_tokens / duration_time if duration_time > 0 else float('inf')
    print(f"requests: {num_requests}, QPS: {qps:.2f}, tokens: {num_tokens}, Token/s: {token_per_sec:.2f}")
