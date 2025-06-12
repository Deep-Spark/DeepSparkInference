# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import yaml
import subprocess
import json
import re
import time
import logging
import os
import sys
import argparse

import utils

# 配置日志
debug_level = logging.DEBUG if utils.is_debug() else logging.INFO
logging.basicConfig(
    handlers=[logging.FileHandler("output.log"), logging.StreamHandler()],
    level=debug_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

METRIC_PATTERN = r"{'metricResult':.*}"

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, help="model name, e.g: alexnet")
    args = parser.parse_args()

    if args.model:
        test_model = args.model
    else:
        test_model = os.environ.get("TEST_CASE")
    logging.info(f"Test case to run: {test_model}")
    if not test_model:
        logging.error("test model case is empty")
        sys.exit(-1)
    
    model = get_model_config(test_model)
    if not model:
        logging.error("mode config is empty")
        sys.exit(-1)

    result = {}
    # NLP模型
    if model["category"] in ["nlp/llm"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_nlp_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    logging.info(f"Full text result: {result}")

def get_model_config(mode_name):
    with open("model_info.json", mode='r', encoding='utf-8') as file:
        models = json.load(file)

    for model in models['models']:
        if model["model_name"] == mode_name.lower() and (model["framework"] == "trtllm" or model["framework"] == "tgi"):
            return model
    return

def check_model_result(result):
    status = "PASS"
    for prec in ["fp16", "int8"]:
        if prec in result["result"]:
            if result["result"][prec]["status"] == "FAIL":
                status = "FAIL"
                break
    result["status"] = status

def run_nlp_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    set -x
    cd ../{model['model_path']}
    bash ci/prepare.sh
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        set -x
        cd ../{model['model_path']}
        """
        if model_name == "llama2-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            bash scripts/test_trtllm_llama2_7b_gpu1_build.sh
            bash scripts/test_trtllm_llama2_7b_gpu1.sh
            """
        elif model_name == "llama2-13b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1
            bash scripts/test_trtllm_llama2_13b_gpu2_build.sh
            bash scripts/test_trtllm_llama2_13b_gpu2.sh
            """
        elif model_name == "llama2-70b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            bash scripts/test_trtllm_llama2_70b_gpu8_build.sh
            bash scripts/test_trtllm_llama2_70b_gpu8.sh
            """
        elif model_name == "qwen1.5-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=1
            python3 offline_inference.py --model2path ./data/Qwen1.5-7B
            """

        r, t = run_script(script)
        sout = r.stdout

        if model_name == "qwen1.5-7b":
            pattern = r"generate length: (\d+)"
            generate_length_match  = re.search(pattern, sout)
            result["result"].setdefault(prec, {"status": "FAIL"})
            if generate_length_match:
                generate_length = float(generate_length_match.group(1))
                result["result"][prec] = {"generate_length": generate_length}
                result["result"][prec]["status"] = "PASS"
            pattern = r"qps: (\d+)"
            qps_match = re.search(pattern, sout)
            if qps_match:
                qps = float(qps_match.group(1))
                result["result"][prec] = {"qps": qps}
                result["result"][prec]["status"] = "PASS"
        else:
            # TensorRT-LLM (total latency: 49.15709829330444 sec)
            # TensorRT-LLM (total output tokens: 1788)
            # TensorRT-LLM (tokens per second: 36.37318031531448)
            pattern = r"\(total latency: (\d+\.\d+) sec\)"
            latency_match = re.search(pattern, sout)
            result["result"].setdefault(prec, {"status": "FAIL"})
            if latency_match:
                latency = float(latency_match.group(1))
                result["result"][prec] = {"latency": latency}
                result["result"][prec]["status"] = "PASS"

            pattern = r"\(total output tokens: (\d+)\)"
            tokens_match = re.search(pattern, sout)
            if tokens_match:
                tokens = int(tokens_match.group(1))
                result["result"][prec]["tokens"] = tokens

            pattern = r"\(tokens per second: (\d+\.\d+)\)"
            tokens_per_second_match = re.search(pattern, sout)
            if tokens_per_second_match:
                tokens_per_second = float(tokens_per_second_match.group(1))
                result["result"][prec]["tokens_per_second"] = tokens_per_second

        result["result"][prec]["Cost time (s)"] = t
    return result

def get_metric_result(str):
    if str:
        return json.loads(str.replace("'", "\""))["metricResult"]
    return None

def run_script(script):
    start_time = time.perf_counter()
    result = subprocess.run(
        script, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logging.debug(f"执行命令：\n{script}")
    logging.debug("执行时间: {:.4f} 秒".format(execution_time))
    logging.debug(f"标准输出: {result.stdout}")
    logging.debug(f"标准错误: {result.stderr}")
    logging.debug(f"返回码: {result.returncode}")
    return result, execution_time

if __name__ == "__main__":
    main()
