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

from copy import deepcopy
from typing import Tuple, List, Union

import codecs
import logging
import argparse

# 对于chat模型，或者模型需要特定的输入，需要对prompt进行额外的处理。
# 如果您在使用中有额外的prompt处理方式需求或者错误反馈，可以联系王坚或者巩亚飞，我们会对modelzoo进行更新适配。

def sampling_add_cli_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    args.add_argument(
        '--n',
        type=int,
        default=1,
        help="Number of output sequences to return for the given prompt.")
    args.add_argument(
        '--best-of',
        type=int,
        default=None,
        help="Number of output sequences that are generated from the prompt. "
        "From these `best_of` sequences, the top `n` sequences are returned. "
        "`best_of` must be greater than or equal to `n`. This is treated as "
        "the beam width when `use_beam_search` is True. By default, `best_of`"
        "is set to `n`.")
    args.add_argument(
        '--presence-penalty',
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on whether they "
        "appear in the generated text so far. Values > 0 encourage the model "
        "to use new tokens, while values < 0 encourage the model to repeat "
        "tokens.")
    args.add_argument(
        '--frequency-penalty',
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on their "
        " frequency in the generated text so far. Values > 0 encourage the "
        " model to use new tokens, while values < 0 encourage the model to "
        "repeat tokens.")
    args.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help="Float that penalizes new tokens based on whether "
        "they appear in the prompt and the generated text so far. Values > 1 "
        "encourage the model to use new tokens, while values < 1 encourage "
        "the model to repeat tokens.")
    args.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help="Float that controls the randomness of the sampling. Lower "
        "values make the model more deterministic, while higher values make "
        "the model more random. Zero means greedy sampling.")
    args.add_argument(
        '--top-p',
        type=float,
        default=1.0,
        help="Float that controls the cumulative probability of the top tokens "
            "to consider. Must be in (0, 1]. Set to 1 to consider all tokens.")
    args.add_argument(
        '--top-k',
        type=int,
        default=-1,
        help="Integer that controls the number of top tokens to consider. Set "
        "to -1 to consider all tokens.")
    args.add_argument(
        '--min-p',
        type=float,
        default=0.0,
        help="Float that represents the minimum probability for a token to be "
        "considered, relative to the probability of the most likely token. "
        "Must be in [0, 1]. Set to 0 to disable this.")
    args.add_argument(
        '--use-beam-search',
        default=False,
        action="store_true",
        help="Whether to use beam search instead of sampling.")
    args.add_argument(
        '--length-penalty',
        type=float,
        default=1.0,
        help="Float that penalizes sequences based on their length. Used in beam search.")
    args.add_argument(
        '--stop',
        type=str,
        default=None,
        help="List of strings that stop the generation when they are generated. "
        "The returned output will not contain the stop strings.")
    args.add_argument(
        '--stop-token-ids',
        type=int,
        default=None,
        help="List of tokens that stop the generation when they are "
        "generated. The returned output will contain the stop tokens unless "
        "the stop tokens are special tokens.")
    args.add_argument(
        '--include-stop-str-in-output',
        default=False,
        action="store_true",
        help="Whether to include the stop strings in output text. Defaults to False.")
    args.add_argument(
        '--ignore-eos',
        default=False,
        action="store_true",
        help="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.")
    args.add_argument(
        '--max-tokens',
        type=int,
        default=16,
        help="Maximum number of tokens to generate per output sequence.")
    args.add_argument(
        '--logprobs',
        type=int,
        default=None,
        help="NNumber of log probabilities to return per output token. "
        "Note that the implementation follows the OpenAI API: The return "
        "result includes the log probabilities on the `logprobs` most likely "
        "tokens, as well the chosen tokens. The API will always return the "
        "log probability of the sampled token, so there  may be up to "
        "`logprobs+1` elements in the response.")
    args.add_argument(
        '--prompt-logprobs',
        type=int,
        default=None,
        help="Number of log probabilities to return per prompt token.")
    args.add_argument(
        '--skip-special-tokens',
        default=True,
        action="store_false",
        help="Whether to skip special tokens in the output.")
    args.add_argument(
        '--spaces-between-special-tokens',
        default=True,
        action="store_false",
        help="Whether to add spaces between special tokens in the output.  Defaults to True.")
    # early_stopping logits_processors seed
    return args


def load_chat_template(tokenizer, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logging.info(
                f"Using supplied chat template:\n{tokenizer.chat_template}"
            )
        elif tokenizer.chat_template is not None:
            logging.info(
                f"Using default chat template:\n{tokenizer.chat_template}"
            )
        else:
            logging.warning(
                "No chat template provided. Chat API will not work.")

def default_build_chat(tokenizer,prompt):
    return prompt

def chatglm2_build_chat(tokenizer,prompt):
    return tokenizer.build_prompt(prompt)

def chatglm3_build_chat(tokenizer,prompt):
    return tokenizer.build_chat_input(prompt).input_ids[0].tolist()

def llama2_build_chat(tokenizer,prompt):
    return f"[INST]{prompt}[/INST]"

# adapt from https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/generation_utils.py
def baichuan2_build_chat(tokenizer, prompt, max_new_tokens=512):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    messages = [{"role": "user", "content": f"{prompt}"}]
    max_new_tokens = max_new_tokens
    max_input_tokens = 4096 - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(196)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens

def qwen_build_chat(
    tokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def codellama_build_chat(tokenizer,prompt):
    return "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:{}[/INST]".format(prompt)

def build_chat(tokenizer, prompt, model_name, **kwargs):
    model_name = model_name.lower()
        # return str or list[int]
    if "chatglm2" in model_name:
        prompt = chatglm2_build_chat(tokenizer,prompt)
    elif "chatglm3" in model_name:
        prompt = chatglm3_build_chat(tokenizer,prompt)
    elif "llama2" in model_name and 'chat' in model_name:
        prompt = llama2_build_chat(tokenizer,prompt)
    elif "baichuan2" in model_name and 'chat' in model_name:
        prompt = baichuan2_build_chat(tokenizer,prompt, kwargs['max_length'])
    elif "qwen" in model_name and 'chat' in model_name:
        prompt = qwen_build_chat(tokenizer,prompt)
    elif "code" in model_name and 'llama' in model_name:
        prompt = codellama_build_chat(tokenizer,prompt)
    else:
        prompt = default_build_chat(tokenizer,prompt)
    return prompt


# for output
def default_post_process(output):
    return output

def glm2_post_process(output):
    output = output.strip()
    output = output.replace("[[训练时间]]", "2023年")
    return output

def glm3_post_process(output, history=[]):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content

def post_process(response, model_name,**kwargs):
    model_name = model_name.lower()
    if "chatglm2" in model_name:
        response = glm2_post_process(response)
    elif "chatglm3" in model_name:
        response = glm3_post_process(response)
    else:
        response = default_post_process(response)
    return response