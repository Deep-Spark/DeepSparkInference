#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import copy
import json
import fire
import requests
from loguru import logger
from tqdm import tqdm


supported_dataset = {
    "MMMU_BETA": "MMMU_BETA.json",
    "MMSTAR_BETA": "MMSTAR_BETA.json",
}


def move_question_to_front(data):
    text_items = [item for item in data if item.get("type") == "text"]
    other_items = [item for item in data if item.get("type") != "text"]
    return text_items + other_items


def batch_processing(func, arg_list, num_workers=16):
    all_data = []
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
  
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(func, arg) for arg in arg_list]
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                all_data.append(result)
  
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e
    finally:
        return all_data


def load_dataset(dataset_name):
    assert dataset_name in supported_dataset, f"Dataset {dataset_name} not supported"
    logger.info(f"Loading dataset {dataset_name}")
    dataset = json.load(open(supported_dataset[dataset_name]))
    return dataset

    
def process_messages(messages):
    messages = copy.deepcopy(messages)
    # drop ground truth
    if messages[-1]['role'].lower() == 'assistant':
        messages.pop(-1)
    return messages


def forward(model, history, infer_params, base_url, timeout, retry_times=10):
    data = {
        "model": model,
        "messages": history,
        **infer_params,
    }
    resp = requests.post(base_url, json=data, timeout=timeout, proxies={"http": [], "https": []})
    if resp.status_code != 200:
        logger.warning(f"Error: [{resp.status_code}] {resp.text}, retry {retry_times} times")
        if retry_times > 0:
            return forward(model, history, infer_params, base_url, timeout, retry_times - 1)
        else:
            raise Exception(f"Error: [{resp.status_code}] {resp.text}")
    else:
        return resp.json()['choices'][0]['message']['content']


def post_process(raw_output: str) -> str:
    model_ans = raw_output.split("</think>")[-1].strip()
    model_ans = model_ans.split('answer')[-1].strip()
    model_ans = model_ans.split('Answer')[-1].strip()
    
    if '\\boxed{' in model_ans:
        matches = re.findall(r'\\boxed\{(.*?)\}', model_ans)
        if matches:
            model_ans = matches[-1]

    extracted_ans = re.findall(r'[A-Z]', model_ans)
    if len(extracted_ans) > 0:
        model_ans = extracted_ans[0]
    else:
        model_ans = model_ans
    
    return model_ans


def is_correct(model_ans: str, gt_ans: str) -> bool:
    if len(str(gt_ans.strip())) == 1 and str(gt_ans.strip()) >= 'A' and str(gt_ans.strip()) <= 'Z':
        pass
    else:
        gt_ans = "A"

    if gt_ans.lower() == model_ans.lower():
        return True
    else:
        return False


def inference_one_sample(args):
    model, data, infer_params, base_url, timeout = args
    messages = process_messages(data)
    model_ans = forward(model, messages, infer_params, base_url, timeout)
    model_ans = post_process(model_ans)
    score = is_correct(model_ans, data[-1]['content'])
    return model_ans, score


def main(dataset_name, model, ip, port, timeout=3600, num_workers=16, output_path=None):
    infer_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 28672,
        "seed":42
    }

    base_url = f"http://{ip}:{port}/v1/chat/completions"

    if not output_path:
        output_path = f"{model}_{dataset_name}.json"

    dataset = load_dataset(dataset_name)

    task_args = []
    for data in dataset:
        if dataset_name =="MMSTAR_BETA":
            data[0]["content"] = move_question_to_front(data[0]["content"]) 
        task_args.append([model, data, infer_params, base_url, timeout])

    logger.info(f"begin to inference")
    results = batch_processing(inference_one_sample, task_args, num_workers=num_workers)

    raw_output_list = []
    score_list = []
    for result in results:
        raw_output_list.append(result[0])
        score_list.append(result[1])

    total_score = sum(score_list) / len(score_list)

    logger.info(f"Inference finished, Total score: {total_score}")
    logger.info(f"Inference samples: {len(results)}, total samples: {len(dataset)}")
    logger.info(f"Saving results to {output_path}")

    with open(output_path, 'w') as f:
        json.dump({
            "total_score": total_score,
            "raw_output_list": raw_output_list,
            "score_list": score_list,
        }, f, indent=4, ensure_ascii=False)

        
if __name__ == "__main__":
    fire.Fire(main)