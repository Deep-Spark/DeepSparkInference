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
    if model["task_type"] == "cv/classification":
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None and (d_url.endswith(".pth") or d_url.endswith(".pt")):
            result = run_clf_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # 检测模型
    if model["task_type"] in ["cv/detection", "cv/pose_estimation"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None and (d_url.endswith(".pth") or d_url.endswith(".pt") or d_url.endswith(".weights")):
            result = run_detec_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # OCR模型
    if model["task_type"] in ["cv/ocr"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_ocr_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # Trace模型
    if model["task_type"] in ["cv/trace"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_trace_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # Speech模型
    if model["task_type"] in ["speech/speech_recognition"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_speech_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # NLP模型
    if model["task_type"] in ["nlp/language_model"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_nlp_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    logging.info(f"Full text result: {result}")

def get_model_config(mode_name):
    with open("models_igie.yaml", "r") as file:
        models = yaml.safe_load(file)

    for model in models:
        if model["name"] == mode_name:
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
        pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=\s?\d+\s?\] =\s*([\d.]+)"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {})
            result["result"][prec] = result["result"][prec] | {m[0]: m[1]}
        if matchs and len(matchs) == 2:
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")

    return result

def run_ocr_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/igie/{dataset_n} ./
    unzip /mnt/deepspark/repos/PaddleOCR-release-2.6.zip -d ./PaddleOCR
    bash ci/prepare.sh
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['relative_path']}
        export DATASETS_DIR=./{dataset_n}/
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

        pattern = METRIC_PATTERN
        matchs = re.findall(pattern, sout)
        if matchs and len(matchs) == 1:
            result["result"].setdefault(prec, {})
            result["result"][prec].update(get_metric_result(matchs[0]))
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")

    return result

def run_trace_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/igie/{dataset_n} ./
    """

    if model["need_third_part"]:
        prepare_script += "unzip /mnt/deepspark/repos/fast-reid.zip -d ./fast-reid\n"

    prepare_script += """
    bash ci/prepare.sh
    ls -l | grep onnx
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['relative_path']}
        export DATASETS_DIR=./{dataset_n}/
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
        pattern = METRIC_PATTERN
        matchs = re.findall(pattern, sout)
        if matchs and len(matchs) == 1:
            result["result"].setdefault(prec, {})
            result["result"][prec].update(get_metric_result(matchs[0]))
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")
    return result

# BERT series models
def run_nlp_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    target_dirs = {"bert_base_squad": "csarron/bert-base-uncased-squad-v1", "bert_base_ner":"test"}
    target_dir = target_dirs[model_name]
    dirname = os.path.dirname(target_dir)
    mkdir_script = f"mkdir -p {dirname}" if dirname else ""

    prepare_script = f"""
    set -x
    cd ../{model['relative_path']}
    {mkdir_script}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./{target_dir}
    export DATASETS_DIR=/mnt/deepspark/data/datasets/igie/{dataset_n}
    bash ci/prepare.sh
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        set -x
        export DATASETS_DIR=/mnt/deepspark/data/datasets/igie/{dataset_n}
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        r, t = run_script(script)
        sout = r.stdout

        pattern = METRIC_PATTERN
        matchs = re.findall(pattern, sout)
        result["result"].setdefault(prec, {"status": "FAIL"})
        logging.debug(f"matchs:\n{matchs}")
        for m in matchs:
            result["result"][prec].update(get_metric_result(m))
        if len(matchs) == 2:
            result["result"][prec]["status"] = "PASS"

        result["result"][prec]["Cost time (s)"] = t
    return result

def run_speech_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/igie/{dataset_n} ./
    """

    if model["need_third_part"]:
        prepare_script += "unzip /mnt/deepspark/repos/kenlm.zip -d ./ctc_decoder/swig/kenlm\n"
        prepare_script += "unzip /mnt/deepspark/repos/ThreadPool.zip -d ./ctc_decoder/swig/ThreadPool\n"
        prepare_script += "tar -xzvf /mnt/deepspark/repos/openfst-1.6.3.tar.gz -C ./ctc_decoder/swig/\n"

    prepare_script += """
    export PYTHONPATH=`pwd`/wenet:$PYTHONPATH
    echo $PYTHONPATH
    bash ci/prepare.sh
    ls -l | grep onnx
    """
    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['relative_path']}
        export PYTHONPATH=./wenet:$PYTHONPATH
        echo $PYTHONPATH
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
        pattern = METRIC_PATTERN
        matchs = re.findall(pattern, sout)
        if matchs and len(matchs) == 1:
            result["result"].setdefault(prec, {})
            result["result"][prec].update(get_metric_result(matchs[0]))
            result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")
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
