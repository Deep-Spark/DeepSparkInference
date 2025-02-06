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
import sys
from pathlib import Path

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

    engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
    sampling_args = [
        param.name
        for param in list(
            inspect.signature(SamplingParams).parameters.values()
        )[1:]
    ]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    sampling_params = {
        attr: getattr(args, attr) for attr in sampling_args if args.__contains__(attr)
    }

    model_name = args.model.strip()
    model_name = model_name if args.model[-1] != "/" else model_name[:-1]
    model_name = model_name.rsplit("/")[-1]

    # Sample prompts.
    prompts = [
        "Shanghai is one of the most prosperous cities in China, with a GDP of over $300 billion. Shanghai has the fastest growing economy in China and is the second busiest port in the world. In addition to being a hub for business, Shanghai is also a major tourist destination. It is known for its diverse culture and many historical sites.\nThe city of Shanghai is located on the coast of the Pacific Ocean in east-central China. It is bordered by Jiangsu Province to the north, Zhejiang Province to the south, and Jiangsu Province to the west.",
        "What signs may indicate that a person is experiencing anxiety?",
        "Describe how to make cheese pizza.",
        "Write a review article on the development of 5G networks.",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)

    # Create an LLM.
    llm = LLM(**engine_params)

    # process chat template
    if args.remove_chat_template:
        prompts_new = prompts
        if "chat" in model_name.lower():
            logging.warning(
                f"The model name from model path is {model_name}, so we guess you are using the chat model and the additional processing is required for the input prompt. "
                f"If the result is not quite correct, please ensure you do not pass --remove_chat_template in CLI."
            )
    else:
        # Build chat model promopt
        # logging.warning("If you are using a non chat model, please pass the --remove_chat_template in CLI.")
        # Try use transformers's apply_chat_template, if chat_template is None, will use defalut template.
        # For some old models, the default template may cause bad answers. we don't consider this situation,
        # because the Transformers team is advancing the chat template. For more informatino about it,
        # please refer to https://huggingface.co/docs/transformers/main/chat_templating
        try:
            load_chat_template(llm.get_tokenizer(), args.chat_template)
            prompts_new = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = llm.get_tokenizer().apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts_new.append(text)
        except:
            logging.warning(
                "use tokenizer apply_chat_template function failed, may because of low transformers version...(try use transformers>=4.34.0)"
            )
            prompts_new = prompts

    outputs = (
        llm.generate(prompts_new, sampling_params, use_tqdm=False)
        if isinstance(prompts_new[0], str)
        else llm.generate(
            sampling_params=sampling_params,
            prompt_token_ids=prompts_new,
            use_tqdm=False,
        )
    )
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    outputs = (
        llm.generate(prompts_new, sampling_params)
        if isinstance(prompts_new[0], str)
        else llm.generate(sampling_params=sampling_params, prompt_token_ids=prompts_new)
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time

    num_tokens = 0
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = prompts[i]  # show the origin prompt. actully prompt is "output.prompt"
        generated_text = output.outputs[0].text

        num_tokens += len(output.outputs[0].token_ids)
        print(f"Prompt: {prompt}\nGenerated text: {generated_text} \n")
    print(f"tokens: {num_tokens}, QPS: {num_tokens/duration_time}")
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["tokens"] = num_tokens
    metricResult["metricResult"]["QPS"] = round(num_tokens/duration_time,3)
    print(metricResult)
