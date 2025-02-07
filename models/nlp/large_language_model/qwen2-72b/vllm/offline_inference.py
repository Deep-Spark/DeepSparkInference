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
        )
    ]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    sampling_params = {
        attr: getattr(args, attr) for attr in sampling_args if args.__contains__(attr)
    }

    model_name = os.path.dirname(args.model).rsplit("/")[-1]

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
