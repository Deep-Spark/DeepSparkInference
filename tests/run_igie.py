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
    if model["category"] in ["cv/classification", "cv/semantic_segmentation"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_clf_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # 检测模型
    if model["category"] in ["cv/object_detection", "cv/pose_estimation"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_detec_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # OCR模型
    if model["category"] in ["cv/ocr"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_ocr_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # Trace模型
    if model["category"] in ["cv/trace"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_trace_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # multi_object_tracking模型
    if model["category"] in ["cv/multi_object_tracking"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_multi_object_tracking_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # Speech模型
    if model["category"] in ["audio/speech_recognition"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_speech_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    # NLP模型
    if model["category"] in ["nlp/plm"]:
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
        if model["model_name"] == mode_name.lower() and model["framework"] == "igie":
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
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    """
    if model["category"] == "cv/semantic_segmentation":
        prepare_script += """
        pip install /mnt/deepspark/install/mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
        """
    prepare_script += f"""
    bash ci/prepare.sh
    ls -l | grep onnx
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        if model_name == "unet":
            script = f"""
            export DATASETS_DIR=/mnt/deepspark/data/datasets/{dataset_n}
            export RUN_DIR=../../igie_common/
            cd ../{model['model_path']}
            bash scripts/infer_{model_name}_{prec}_accuracy.sh
            bash scripts/infer_{model_name}_{prec}_performance.sh
            """
        else:
            script = f"""
            export DATASETS_DIR=/mnt/deepspark/data/datasets/imagenet-val
            export RUN_DIR=../../igie_common/
            cd ../{model['model_path']}
            bash scripts/infer_{model_name}_{prec}_accuracy.sh
            bash scripts/infer_{model_name}_{prec}_performance.sh
            """

        r, t = run_script(script)
        sout = r.stdout
        pattern = r"\* ([\w\d ]+):\s*([\d.]+)[ ms%]*, ([\w\d ]+):\s*([\d.]+)[ ms%]*"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {"status": "FAIL"})
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
                result["result"][prec] = result["result"][prec] | {m[0]: m[1], m[2]: m[3]}
        if matchs:
            if len(matchs) == 2:
                result["result"][prec]["status"] = "PASS"
            else:
                # Define regex pattern to match key-value pairs inside curly braces
                kv_pattern = r"'(\w+)'\s*:\s*([\d.]+)"
                # Find all matches
                kv_matches = re.findall(kv_pattern, sout)
                for key, value in kv_matches:
                    result["result"][prec]["status"] = "PASS"
                    try:
                        result["result"][prec][key] = float(value)
                    except ValueError:
                        print("The string cannot be converted to a float.")
                        result["result"][prec][key] = value

        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")
    return result

def run_detec_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/{dataset_n} ./
    """
    # for 4.3.0 sdk need pre install mmcv
    prepare_script += """
    pip install /mnt/deepspark/install/mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
    """

    # if model["need_third_part"] and model["3rd_party_repo"]:
    #     third_party_repo = model["3rd_party_repo"]
    #     prepare_script += f"unzip /mnt/deepspark/data/3rd_party/{third_party_repo}.zip -d ./\n"
    prepare_script += "bash ci/prepare.sh\n"

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['model_path']}
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
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
                result["result"][prec] = result["result"][prec] | {m[0]: m[1], m[2]: m[3]}
        pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=\s?\d+\s?\] =\s*([\d.]+)"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {})
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1])}
            except ValueError:
                print("The string cannot be converted to a float.")
                result["result"][prec] = result["result"][prec] | {m[0]: m[1]}
        if matchs and len(matchs) == 2:
            result["result"][prec]["status"] = "PASS"
        else:
            pattern = METRIC_PATTERN
            matchs = re.findall(pattern, sout)
            if matchs and len(matchs) == 1:
                result["result"].setdefault(prec, {})
                result["result"][prec].update(get_metric_result(matchs[0]))
                result["result"][prec]["status"] = "PASS"
        result["result"][prec]["Cost time (s)"] = t
        logging.debug(f"matchs:\n{matchs}")

    return result

def run_ocr_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    # for 4.3.0 sdk need pre install paddle
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/{dataset_n} ./
    pip install /mnt/deepspark/install/paddlepaddle-3.0.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
    unzip -q /mnt/deepspark/data/3rd_party/PaddleOCR-release-2.6.zip -d ./PaddleOCR
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
        cd ../{model['model_path']}
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
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
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
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/{dataset_n} ./
    """

    if model["need_third_part"]:
        prepare_script += "unzip -q /mnt/deepspark/data/3rd_party/fast-reid.zip -d ./fast-reid\n"

    prepare_script += """
    bash ci/prepare.sh
    ls -l | grep onnx
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['model_path']}
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
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
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

def run_multi_object_tracking_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/{dataset_n} ./
    """

    prepare_script += """
    bash ci/prepare.sh
    ls -l | grep onnx
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['model_path']}
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
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
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
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    target_dirs = {"bert_base_squad": "csarron/bert-base-uncased-squad-v1", "bert_base_ner":"test", "bert_large_squad": "neuralmagic/bert-large-uncased-finetuned-squadv1"}
    target_dir = target_dirs[model_name]
    dirname = os.path.dirname(target_dir)
    mkdir_script = f"mkdir -p {dirname}" if dirname else ""

    prepare_script = f"""
    set -x
    cd ../{model['model_path']}
    {mkdir_script}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./{target_dir}
    export DATASETS_DIR=/mnt/deepspark/data/datasets/{dataset_n}
    bash ci/prepare.sh
    """

    # prepare int8 model for bert_large_squad
    if model_name == "bert_large_squad":
        prepare_script += "ln -s /mnt/deepspark/data/checkpoints/bert_large_int8.hdf5 ./\n"

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        set -x
        export DATASETS_DIR=/mnt/deepspark/data/datasets/{dataset_n}
        cd ../{model['model_path']}
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
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "result": {},
    }
    d_url = model["download_url"]
    checkpoint_n = d_url.split("/")[-1]
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./
    ln -s /mnt/deepspark/data/datasets/{dataset_n} ./
    """

    if model["need_third_part"] and model_name == "conformer":
        prepare_script += "unzip -q /mnt/deepspark/data/3rd_party/kenlm.zip -d ./ctc_decoder/swig/kenlm\n"
        prepare_script += "unzip -q /mnt/deepspark/data/3rd_party/ThreadPool.zip -d ./ctc_decoder/swig/ThreadPool\n"
        prepare_script += "tar -xzvf /mnt/deepspark/data/3rd_party/openfst-1.6.3.tar.gz -C ./ctc_decoder/swig/\n"

    prepare_script += """
    export PYTHONPATH=`pwd`/wenet:$PYTHONPATH
    echo $PYTHONPATH
    bash ci/prepare.sh
    ls -l | grep onnx
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['model_path']}
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
            try:
                result["result"][prec] = result["result"][prec] | {m[0]: float(m[1]), m[2]: float(m[3])}
            except ValueError:
                print("The string cannot be converted to a float.")
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
