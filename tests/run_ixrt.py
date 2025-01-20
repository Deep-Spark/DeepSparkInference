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
    if model["task_type"] == "cv/classification":
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_clf_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # 检测模型
    if model["task_type"] in ["cv/detection", "cv/pose_estimation"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_detec_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    # Segmentation模型
    if model["task_type"] in ["cv/segmentation", "cv/face", "multimodal/text_and_image"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_segmentation_and_face_testcase(model)
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
    if model["task_type"] in ["nlp/language_model", "recommendation/ctr-prediction"]:
        logging.info(f"Start running {model['name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_nlp_testcase(model)
            check_model_result(result)
            logging.debug(f"The result of {model['name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['name']} test case.")

    logging.info(f"Full text result: {result}")

def get_model_config(mode_name):
    with open("models_ixrt.yaml", "r") as file:
        models = yaml.safe_load(file)

    for model in models:
        if model["name"] == mode_name.lower():
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
    bash ci/prepare.sh
    """
    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    config_name = model_name.upper()

    patterns = {
        "FPS": r"FPS\s*:\s*(\d+\.?\d*)",
        "Acc1": r"Acc@1\s*:\s*(\d+\.?\d*)",
        "Acc5": r"Acc@5\s*:\s*(\d+\.?\d*)",
        "E2E": r"E2E time\s*:\s*(\d+\.\d+)"
    }

    combined_pattern = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in patterns.items()))

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['relative_path']}
        export DATASETS_DIR=/root/data/datasets/imagenet-val
        export PROJ_DIR=./
        export CHECKPOINTS_DIR=./checkpoints
        export RUN_DIR=./
        export CONFIG_DIR=config/{config_name}_CONFIG
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        if model_name == "swin_transformer_large":
            script = f"""
            cd ../{model['relative_path']}
            export ORIGIN_ONNX_NAME=./swin-large-torch-fp32
            export OPTIMIER_FILE=/root/data/3rd_party/iluvatar-corex-ixrt/tools/optimizer/optimizer.py
            export PROJ_PATH=./
            bash scripts/infer_swinl_fp16_performance.sh
            cd ./ByteMLPerf/byte_infer_perf/general_perf
            python3 core/perf_engine.py --hardware_type ILUVATAR --task swin-large-torch-fp32
            """

        r, t = run_script(script)
        sout = r.stdout
        matchs = combined_pattern.finditer(sout)
        result["result"].setdefault(prec, {"status": "FAIL"})
        match_count = 0
        for match in matchs:
            for name, value in match.groupdict().items():
                if value:
                    match_count += 1
                    result["result"][prec][name] = float(f"{float(value.split(':')[1].strip()):.3f}")
                    break

        if match_count == len(patterns):
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
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    ln -s /root/data/datasets/{dataset_n} ./
    bash ci/prepare.sh
    """

    # add pip list info when in debug mode
    if utils.is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    run_script(prepare_script)

    config_name = model_name.upper()

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = f"""
        cd ../{model['relative_path']}
        export DATASETS_DIR=./{dataset_n}/

        export MODEL_PATH=./{model_name}.onnx

        export PROJ_DIR=./
        export CHECKPOINTS_DIR=./checkpoints
        export COCO_GT=./{dataset_n}/annotations/instances_val2017.json
        export EVAL_DIR=./{dataset_n}/val2017
        export RUN_DIR=./
        export CONFIG_DIR=config/{config_name}_CONFIG

        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        if model_name == "rtmpose":
            script = f"""
                cd ../{model['relative_path']}
                python3 predict.py --model data/rtmpose/rtmpose_opt.onnx --precision fp16 --img_path demo/demo.jpg
                """

        r, t = run_script(script)
        sout = r.stdout
        fps_pattern = r"(?P<FPS>FPS\s*:\s*(\d+\.?\d*))"
        e2e_pattern = r"(?P<E2E>\s*E2E time\s*:\s*(\d+\.\d+)\s)"
        combined_pattern = re.compile(f"{fps_pattern}|{e2e_pattern}")
        matchs = combined_pattern.finditer(sout)
        for match in matchs:
            result["result"].setdefault(prec, {"status": "FAIL"})
            for name, value in match.groupdict().items():
                if value:
                    try:
                        result["result"][prec][name] = float(f"{float(value.split(':')[1].strip()):.3f}")
                        break
                    except ValueError:
                        print("The string cannot be converted to a float.")
                        result["result"][prec][name] = value
        pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=\s?\d+\s?\] =\s*([\d.]+)"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result["result"].setdefault(prec, {})
            try:
                result["result"][prec][m[0]] = float(m[1])
            except ValueError:
                print("The string cannot be converted to a float.")
                result["result"][prec][m[0]] = m[1]
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

def run_segmentation_and_face_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    dataset_n = model["datasets"].split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
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
        cd ../{model['relative_path']}
        export DATASETS_DIR=./{dataset_n}/
        export PROJ_DIR=./
        export CHECKPOINTS_DIR=./checkpoints
        export COCO_GT=./{dataset_n}/annotations/instances_val2017.json
        export EVAL_DIR=./{dataset_n}/val2017
        export RUN_DIR=./

        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        if model_name == "clip":
            script = f"""
            cd ../{model['relative_path']}
            python3 inference.py
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
        logging.debug(f"matchs:\n{matchs}")
    return result

# BERT series models
def run_nlp_testcase(model):
    model_name = model["name"]
    result = {
        "name": model_name,
        "result": {},
    }
    prepare_script = f"""
    set -x
    cd ../{model['relative_path']}
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
        cd ../{model['relative_path']}
        export ORIGIN_ONNX_NAME=./data/open_{model_name}/{model_name}
        export OPTIMIER_FILE=/root/data/3rd_party/iluvatar-corex-ixrt/tools/optimizer/optimizer.py
        export PROJ_PATH=./
        bash scripts/infer_{model_name}_{prec}_performance.sh
        cd ./ByteMLPerf/byte_infer_perf/general_perf
        """
        if model_name == "roformer" or model_name == "widedeep":
            script += f"""
            python3 core/perf_engine.py --hardware_type ILUVATAR --task {model_name}-tf-fp32
            """
        elif model_name == "videobert":
            script += f"""
            python3 core/perf_engine.py --hardware_type ILUVATAR --task {model_name}-onnx-fp32
            """
        else:
            #  model_name == "roberta" or model_name == "deberta" or model_name == "albert"
            script += f"""
            python3 core/perf_engine.py --hardware_type ILUVATAR --task {model_name}-torch-fp32
            """


        if model_name == "bert_base_squad":
            script = f"""
            set -x
            cd ../{model['relative_path']}/python
            bash script/infer_{model_name}_{prec}_ixrt.sh
            """
        elif model_name == "bert_large_squad":
            script = f"""
            set -x
            cd ../{model['relative_path']}/python
            bash script/build_engine.sh --bs 32
            bash script/inference_squad.sh --bs 32
            """
            if prec == "int8":
                script = f"""
                set -x
                cd ../{model['relative_path']}/python
                bash script/build_engine.sh --bs 32 --int8
                bash script/inference_squad.sh --bs 32 --int8
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
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        if model_name == "transformer_asr":
            script = f"""
            cd ../{model['relative_path']}
            python3 inference.py hparams/train_ASR_transformer.yaml --data_folder=/home/data/speechbrain/aishell --engine_path transformer.engine 
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