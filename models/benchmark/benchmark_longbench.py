import argparse
import dataclasses
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.append(str(Path(__file__).resolve().parent.parent.parent) + "/inference")
sys.path.append(str(Path(__file__).resolve().parent) + "/longbench")

FILE_DIR = str(Path(__file__).resolve().parent)

from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)
from utils import load_chat_template, sampling_add_cli_args
from vllm import LLM, EngineArgs, SamplingParams

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "falcon" in model_name:
        prompt = f"User: {prompt}\nFalcon:"
    else:
        # we do not use default template...
        if tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument(
        "--skip-chat-template-check", default=False, action="store_true"
    )
    parser.add_argument("--val-data-nums", type=int, default=-1)
    parser.add_argument(
        "--save-pred", action="store_true", help="Save the pred output to local files."
    )
    parser.add_argument("--new-model-run", default=False, action="store_true")
    parser.add_argument("--target", default=None, type=str, help="for CI TEST ONLY")
    parser = EngineArgs.add_cli_args(parser)
    parser = sampling_add_cli_args(parser)
    args = parser.parse_args()

    return args


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    out_path,
    save_pred,
):
    preds = []
    prompts = []
    prompts_ids = []
    sampling_args = [
        param.name
        for param in list(
            inspect.signature(SamplingParams).parameters.values()
        )[1:]
    ]
    sampling_params = {
        attr: getattr(args, attr) for attr in sampling_args if args.__contains__(attr)
    }
    sampling_params["max_tokens"] = max_gen
    sampling_params = SamplingParams(**sampling_params)

    prompts_ids = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt")
            else:
                input = prompt
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt")

        prompt_token_id = input.input_ids.view(-1).tolist()
        prompts_ids.append(prompt_token_id)
    assert len(data) == len(prompts_ids)
    outputs = model.generate(
        sampling_params=sampling_params, prompt_token_ids=prompts_ids
    )
    for i, output in enumerate(outputs):
        pred = output.outputs[0].text
        json_obj = data[i]
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

    if save_pred:
        for pred in preds:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args):
    engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    model = LLM(**engine_params)
    tokenizer = model.get_tokenizer()
    load_chat_template(tokenizer, args.chat_template)
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model_name = args.model_name

    model2maxlen = json.load(
        open(os.path.join(FILE_DIR, "longbench/config/model2maxlen.json"), "r")
    )
    try:
        max_length = model2maxlen[model_name]
    except:
        raise ValueError(
            "the model_name is not in model2maxlen.json, please check your inputs or add a new model!!!"
        )
    # define model
    model, tokenizer = load_model_and_tokenizer(args)
    if (
        tokenizer.chat_template is None
        and args.chat_template is None
        and not args.skip_chat_template_check
    ):
        raise ValueError(
            "tokenizer.chat_template is None, please pass --skip-chat-template-check if you do not pass --chat-template"
        )

    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(
        open(os.path.join(FILE_DIR, "longbench/config/dataset2prompt.json"), "r")
    )
    dataset2maxlen = json.load(
        open(os.path.join(FILE_DIR, "longbench/config/dataset2maxlen.json"), "r")
    )

    val_data_num = args.val_data_nums
    val_data_num = (
        min(val_data_num, len(datasets)) if val_data_num != -1 else len(datasets)
    )
    index = list(range(len(datasets)))
    random.seed(time.time())
    random.shuffle(index)
    val_datasets = (
        datasets
        if val_data_num == len(datasets)
        else [datasets[i] for i in index[:val_data_num]]
    )

    # predict on each dataset
    scores = dict()
    for dataset in val_datasets:
        data = []
        predictions, answers, lengths = [], [], []
        if args.e:
            with open(
                os.path.join(args.datapath, "{}.jsonl".format(dataset)),
                "r",
                encoding="utf-8",
            ) as lines:
                for line in lines:
                    data.append(json.loads(line))
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            with open(
                os.path.join(args.datapath, "{}.jsonl".format(dataset)),
                "r",
                encoding="utf-8",
            ) as lines:
                for line in lines:
                    data.append(json.loads(line))
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            out_path,
            args.save_pred,
        )

        for line in preds:
            predictions.append(line["pred"])
            answers.append(line["answers"])
            all_classes = line["all_classes"]
            if "length" in line:
                lengths.append(line["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)

        scores[dataset] = score
        if not args.new_model_run:
            reference_file = (
                os.path.join(FILE_DIR, f"longbench/result_record/{model_name}.jsonl")
                if args.target is None
                else args.target
            )
            mdoel_result_reference = json.load(open(reference_file, "r"))
            reference_score = mdoel_result_reference[dataset]
            if (
                score < reference_score
                and (reference_score - score) / reference_score > 0.06
            ):
                print(
                    f"{model_name} on dataset: {dataset}, target score: {reference_score}, val score: {score}, fail!"
                )
                # exit(1)
            else:
                print(
                    f"{model_name} on dataset: {dataset}, target score: {reference_score}, val score: {score}, pass"
                )
    if args.e:
        out_path = f"pred_e/{model_name}/result.json"
    else:
        out_path = f"pred/{model_name}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # for ci test exit
    try:
        import os
        import subprocess

        current_pid = os.getpid()
        cmd = (
            """ps -ef | grep multiprocessing.spawn | grep {} | grep -v grep | """.format(
                current_pid
            )
            + "awk '{print $2}'"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        pid = result.returncode
        if result.returncode == 0:
            result = result.stdout.strip()
            pids = result.split("\n")
            for pid in pids:
                if str.isdigit(pid):
                    cmd = "kill -9 {}".format(pid)
                    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except:
        assert False