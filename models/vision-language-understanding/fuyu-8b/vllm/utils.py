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
import argparse
import codecs
import logging

"""
The following arguments can not be add in args...
early_stopping: Union[bool, str] = False,
early_stopping: Controls the stopping condition for beam search. It
    accepts the following values: `True`, where the generation stops as
    soon as there are `best_of` complete candidates; `False`, where an
    heuristic is applied and the generation stops when is it very
    unlikely to find better candidates; `"never"`, where the beam search
    procedure only stops when there cannot be better candidates
    (canonical beam search algorithm).
stop: Optional[Union[str, List[str]]] = None,
stop_token_ids: Optional[List[int]] = None,
logits_processors: Optional[List[LogitsProcessor]] = None,
logits_processors: List of functions that modify logits based on
    previously generated tokens, and optionally prompt tokens as
    a first argument.
truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
truncate_prompt_tokens: If set to an integer k, will use only the last k
    tokens from the prompt (i.e., left truncation). Defaults to None
    (i.e., no truncation).
    """


def sampling_add_cli_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    args.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of output sequences to return for the given prompt.",
    )
    args.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Number of output sequences that are generated from the prompt. "
        "From these `best_of` sequences, the top `n` sequences are returned. "
        "`best_of` must be greater than or equal to `n`. This is treated as "
        "the beam width when `use_beam_search` is True. By default, `best_of`"
        "is set to `n`.",
    )
    args.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on whether they "
        "appear in the generated text so far. Values > 0 encourage the model "
        "to use new tokens, while values < 0 encourage the model to repeat "
        "tokens.",
    )
    args.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on their "
        " frequency in the generated text so far. Values > 0 encourage the "
        " model to use new tokens, while values < 0 encourage the model to "
        "repeat tokens.",
    )
    args.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Float that penalizes new tokens based on whether "
        "they appear in the prompt and the generated text so far. Values > 1 "
        "encourage the model to use new tokens, while values < 1 encourage "
        "the model to repeat tokens.",
    )
    args.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Float that controls the randomness of the sampling. Lower "
        "values make the model more deterministic, while higher values make "
        "the model more random. Zero means greedy sampling.",
    )
    args.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Float that controls the cumulative probability of the top tokens "
        "to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
    )
    args.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Integer that controls the number of top tokens to consider. Set "
        "to -1 to consider all tokens.",
    )
    args.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Float that represents the minimum probability for a token to be "
        "considered, relative to the probability of the most likely token. "
        "Must be in [0, 1]. Set to 0 to disable this.",
    )
    args.add_argument(
        "--use-beam-search",
        default=False,
        action="store_true",
        help="Whether to use beam search instead of sampling.",
    )
    args.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Float that penalizes sequences based on their length. Used in beam search.",
    )
    args.add_argument(
        "--stop",
        type=str,
        default=None,
        help="List of strings that stop the generation when they are generated. "
        "The returned output will not contain the stop strings.",
    )
    args.add_argument(
        "--stop-token-ids",
        type=int,
        default=None,
        help="List of tokens that stop the generation when they are "
        "generated. The returned output will contain the stop tokens unless "
        "the stop tokens are special tokens.",
    )
    args.add_argument(
        "--include-stop-str-in-output",
        default=False,
        action="store_true",
        help="Whether to include the stop strings in output text. Defaults to False.",
    )
    args.add_argument(
        "--ignore-eos",
        default=False,
        action="store_true",
        help="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    )
    args.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate per output sequence.",
    )
    args.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Minimum number of tokens to generate per output sequence "
        "before EOS or stop_token_ids can be generated",
    )
    args.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="NNumber of log probabilities to return per output token. "
        "Note that the implementation follows the OpenAI API: The return "
        "result includes the log probabilities on the `logprobs` most likely "
        "tokens, as well the chosen tokens. The API will always return the "
        "log probability of the sampled token, so there  may be up to "
        "`logprobs+1` elements in the response.",
    )
    args.add_argument(
        "--prompt-logprobs",
        type=int,
        default=None,
        help="Number of log probabilities to return per prompt token.",
    )
    args.add_argument(
        "--detokenize",
        type=bool,
        default=True,
        help="Whether to detokenize the output. Defaults to True.",
    )
    args.add_argument(
        "--skip-special-tokens",
        default=True,
        action="store_false",
        help="Whether to skip special tokens in the output.",
    )
    args.add_argument(
        "--spaces-between-special-tokens",
        default=True,
        action="store_false",
        help="Whether to add spaces between special tokens in the output.  Defaults to True.",
    )
    return args


def load_chat_template(tokenizer, chat_template):
    if chat_template is not None:
        try:
            with open(chat_template, "r") as f:
                tokenizer.chat_template = f.read()
        except OSError:
            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            tokenizer.chat_template = codecs.decode(chat_template, "unicode_escape")

        logging.info(f"Using supplied chat template:\n{tokenizer.chat_template}")
    elif tokenizer.chat_template is not None:
        logging.info(
            f"Using default chat template:\n{tokenizer.chat_template}. This May lead to unsatisfactory results. You can provide a template.jinja file for vllm."
        )
    else:
        logging.warning(
            "No chat template provided. Chat API will not work. This May lead to unsatisfactory results. You can provide a template.jinja file for vllm."
        )