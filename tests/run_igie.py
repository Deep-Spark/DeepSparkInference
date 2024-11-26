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

# 配置日志
is_debug = os.environ.get("IS_DEBUG")
debug_level = logging.INFO
if is_debug and is_debug.lower()=="true":
    debug_level = logging.DEBUG
logging.basicConfig(
    handlers=[logging.FileHandler("output.log"), logging.StreamHandler()],
    level=debug_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    with open("models_igie.yaml", "r") as file:
        models = yaml.safe_load(file)

    test_cases = os.environ.get("TEST_CASES")
    logging.info(f"Test cases to run: {test_cases}")
    if test_cases:
        avail_models = [tc.strip() for tc in test_cases.split(",")]
    else:
        logging.error("test_cases empty")
        sys.exit(-1)

    test_data = []
    for index, model in enumerate(models):
        # 分类模型
        if model["task_type"] == "cv/classification" and model["name"] in avail_models:
            logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
            d_url = model["download_url"]
            if d_url is not None and (d_url.endswith(".pth") or d_url.endswith(".pt")):
                result = run_clf_testcase(model)
                check_model_result(result)
                test_data.append(result)
                logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
            logging.info(f"End running {model['name']} test case.")

        # 检测模型
        if model["task_type"] == "cv/detection" and model["name"] in avail_models:
            logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
            d_url = model["download_url"]
            if d_url is not None and (d_url.endswith(".pth") or d_url.endswith(".pt") or d_url.endswith(".weights")):
                result = run_detec_testcase(model)
                check_model_result(result)
                test_data.append(result)
                logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
            logging.info(f"End running {model['name']} test case.")

    logging.info(f"Full results:\n{json.dumps(test_data, indent=4)}")

def check_model_result(result):
    status = "PASS"
    for prec in ["fp16", "int8"]:
        if prec in result["result"]:
            if result["result"][prec]["status"] == "FAIL":
                status = "FAIL"
                break
    result["status"] = status

def run_clf_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    bash ci/prepare.sh
    ls -l | grep onnx
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/imagenet-val
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        r, t = run_script(script)
        sout = r.stdout
        pattern = r"\* ([\w\d ]+):\s*([\d.]+)[ ms%]*, ([\w\d ]+):\s*([\d.]+)[ ms%]*"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {"status": "FAIL"})
            result["result"][prec] = result["result"][prec] | {m[0]: m[1], m[2]: m[3]}
        if matchs and len(matchs) == 2:
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")
    return result


def run_detec_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    bash ci/prepare.sh
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/coco
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        r, t = run_script(script)
        sout = r.stdout
        pattern = r"\* ([\w\d ]+):\s*([\d.]+)[ ms%]*, ([\w\d ]+):\s*([\d.]+)[ ms%]*"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {"status": "FAIL"})
            result["result"][prec] = result["result"][prec] | {m[0]: m[1], m[2]: m[3]}
        pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=1000? \] = ([\d.]+)"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {})
            result["result"][prec] = result["result"][prec] | {m[0]: m[1]}
        if matchs and len(matchs) == 2:
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")

    return result


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
