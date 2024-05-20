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
from utils import build_chat,post_process,load_chat_template,sampling_add_cli_args

import logging
import time
import argparse
import dataclasses
import inspect

import torch
from vllm import LLM, SamplingParams, EngineArgs


parser = argparse.ArgumentParser()
parser.add_argument("--chat_template",type=str,default=None)
parser.add_argument("--remove_chat_template",default=True,action="store_false",help="pass this if you are not use a chat model")
parser = EngineArgs.add_cli_args(parser)
parser = sampling_add_cli_args(parser)
args = parser.parse_args()

engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
sampling_args = [param.name for param in list(inspect.signature(SamplingParams.__init__).parameters.values())[1:]]
engine_params = {attr:getattr(args, attr) for attr in engine_args}
sampling_params = {attr:getattr(args, attr) for attr in sampling_args if args.__contains__(attr)}

model_name = args.model.strip()
model_name = model_name if args.model[-1]!='/' else model_name[:-1]
model_name = model_name.rsplit('/')[-1]


# Sample prompts.
prompts = [
            "哪些迹象可能表明一个人正在经历焦虑?", 
            "描述一下如何制作芝士披萨。", 
            "写一篇有关5G网络研发的综述文章。"
           ]

# Create a sampling params object.
sampling_params = SamplingParams(**sampling_params)

# Create an LLM.
llm = LLM(**engine_params)

# process chat template
if not args.remove_chat_template:
    if 'chat' not in model_name.lower():
        logging.warning(f"We assume that you are using the chat model, so additional processing is required for the input prompt. "
                        f"If the result is not quite correct, please ensure that the model path includes the chat character. "
                        f"for now, the model_name from model path is {model_name}")
    prompts_new = prompts
else:
    # Build chat model promopt
    # Try use transformers's apply_chat_template, if chat_template is None, will use defalut template.
    # For some old models, the default template may cause bad answers. we don't consider this situation, 
    # because the Transformers team is advancing the chat template. For more informatino about it, 
    # please refer to https://huggingface.co/docs/transformers/main/chat_templating
    try:
        load_chat_template(llm.get_tokenizer(),args.chat_template)
        prompts_new = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = llm.get_tokenizer().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts_new.append(text)
    except:
        logging.warning("use tokenizer apply_chat_template function failed, may because of low transformers version...(try use transformers>=4.37.0)")
        # Fall back to simple build chat, this part should be controled by model developer, we just provide a simple use cases
        prompts_new = [build_chat(llm.get_tokenizer(),prompt,model_name,max_length=args.max_generate_tokens) for prompt in prompts]

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts_new, sampling_params,use_tqdm=False) if isinstance(prompts_new[0],str) else llm.generate(sampling_params=sampling_params,prompt_token_ids=prompts_new,use_tqdm=False)
torch.cuda.synchronize()

start_time = time.perf_counter()
outputs = llm.generate(prompts_new, sampling_params) if isinstance(prompts_new[0],str) else llm.generate(sampling_params=sampling_params,prompt_token_ids=prompts_new)
torch.cuda.synchronize()
end_time = time.perf_counter()
duration_time = end_time - start_time

num_tokens = 0
# Print the outputs.
for i, output in enumerate(outputs):
    prompt = prompts[i] # show the origin prompt. actully prompt is "output.prompt"
    generated_text = post_process(output.outputs[0].text,model_name)
    
    num_tokens += len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt}\nGenerated text: {generated_text} \n")
print(f"tokens: {num_tokens}, QPS: {num_tokens/duration_time}")

# 0.3.2 tokens: 757, QPS: 97.97229589080902