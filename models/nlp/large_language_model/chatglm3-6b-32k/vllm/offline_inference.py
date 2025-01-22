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
            inspect.signature(SamplingParams.__init__).parameters.values()
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
    prompts = ["哪些迹象可能表明一个人正在经历焦虑?", "描述一下如何制作芝士披萨。", "写一篇有关5G网络研发的综述文章。"]

    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)

    # Create an LLM.
    llm = LLM(**engine_params)

    # process chat template
    if args.remove_chat_template:
        if "chat" in model_name.lower():
            logging.warning(
                f"The model name from model path is {model_name}, so we guess you are using the chat model and the additional processing is required for the input prompt. "
                f"If the result is not quite correct, please ensure you do not pass --remove_chat_template in CLI."
            )
        prompts_new = prompts
    else:
        # Build chat model promopt
        logging.warning(
            "If you are using a non chat model, please pass the --remove_chat_template in CLI."
        )
        logging.warning(
            "For now, openai api chat interface(v1/chat/completions) need you provide a chat template to process prompt(str) for better results. "
            "Otherwise, you have to use the default chat template, which may lead to bad answers. But, the process of building chat input is complex "
            "for some models and the rule of process can not be written as a jinja file. Fortunately, the v1/completions interface support List[int] "
            "params. This means you can process the prompt firstly, then send the List[int] to v1/completions and consider it as v1/chat/completions "
            "to use when you use openai api."
        )
        tokenizer = llm.get_tokenizer()
        prompts_new = []
        for prompt in prompts:
            input_idx = (
                tokenizer.build_chat_input(prompt)["input_ids"][0].cpu().tolist()
            )
            prompts_new.append(input_idx)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
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