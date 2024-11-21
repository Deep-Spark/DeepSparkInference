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

# 配置日志
logging.basicConfig(
    filename="output.log",  # 日志文件名
    level=logging.INFO,     # 日志级别
    format="%(asctime)s - %(message)s"  # 日志格式
)

def main():
    with open("models_igie.yaml", "r") as file:
        models = yaml.safe_load(file)

    sample = {
        'name': 'alexnet',
        'result': {
            'fp16': {
                'Top1 acc':'56.538 %',
                'Top5 acc':'79.055 %',
                'Mean inference time': '1.476 ms',
                'Mean fps': '21685.350'

            },
            'int8': {
                'Top1 acc':'55.538 %',
                'Top5 acc':'78.055 %',
                'Mean inference time': '1.076 ms',
                'Mean fps': '23682'
            }
        },
        'status': 'fail'
    }

    avail_models = ["atss"]
    test_data = []
    # 分类模型
    for index, model in enumerate(models):
        if model["task_type"] == "cv/classification" and model['name'] in avail_models:
            logging.info(f"{index}, {model['name']}")
            logging.info(json.dumps(model, indent=4))
            d_url = model['download_url']
            if d_url is not None and (d_url.endswith('.pth') or d_url.endswith('.pt')):
                test_data.append(run_clf_testcase(model))

            logging.info(json.dumps(test_data, indent=4))

    # 检测模型
    for index, model in enumerate(models):
        avail_models = ["atss"]
        if model["task_type"] == "cv/detection" and model['name'] in avail_models:
            logging.info(f"{index}, {model['name']}")
            logging.info(json.dumps(model, indent=4))
            d_url = model['download_url']
            if d_url is not None and (d_url.endswith('.pth') or d_url.endswith('.pt')):
                test_data.append(run_detec_testcase(model))
            logging.info(json.dumps(test_data, indent=4))

def run_clf_testcase(model):
    model_name = model['name']
    result = {'name': model_name, 'result':{}, }
    d_url = model['download_url']
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    cd ../{model['relative_path']}
    pip3 install onnx tqdm
    python3 export.py --weight /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} --output {model_name}.onnx
    ls -l | grep onnx
    """
    run_script(prepare_script)

    for prec in model['precisions']:
        logging.info(f"run {prec} test case")
        script = f"""
        export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/imagenet-val
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        r = run_script(script)
        sout = r.stdout
        logging.info("标准输出:", r.stdout)
        logging.info("标准错误:", r.stderr)
        logging.info("返回码:", r.returncode)
        pattern = r"\* ([\w\d ]+):\s*([\d.]+)[ ms%]*, ([\w\d ]+):\s*([\d.]+)[ ms%]*"
        # match = re.search(pattern, sout)
        # if match:
        #     result['result'][prec]={match.group(1):match.group(2), match.group(3):match.group(4)}
        matchs = re.findall(pattern, sout)
        for m in matchs:
            # if not result['result'][prec]:
            #     result['result'][prec] = {}
            result['result'].setdefault(prec, {})
            result['result'][prec]=result['result'][prec] | {m[0]:m[1],m[2]:m[3]}
        logging.info("**************")
        logging.info(matchs)
        logging.info("**************")

    return result

def run_detec_testcase(model):
    model_name = model['name']
    result = {'name': model_name, 'result':{}, }
    d_url = model['download_url']
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    apt install -y libgl1-mesa-glx
    cd ../{model['relative_path']}
    cat requirements.txt
    pip3 install -r requirements.txt
    python3 export.py --weight /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} --cfg {model_name}_*_coco.py --output {model_name}.onnx
    ls
    ls -l | grep onnx
    onnxsim {model_name}.onnx {model_name}_opt.onnx
    """
    run_script(prepare_script)

    for prec in model['precisions']:
        logging.info(f"run {prec} test case")
        script = f"""
        export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/coco
        cd ../{model['relative_path']}
        bash scripts/infer_{model_name}_{prec}_accuracy.sh
        # bash scripts/infer_{model_name}_{prec}_performance.sh
        """

        r = run_script(script)
        sout = r.stdout
        pattern = r"\* ([\w\d ]+):\s*([\d.]+)[ ms%]*, ([\w\d ]+):\s*([\d.]+)[ ms%]*"
        # match = re.search(pattern, sout)
        # if match:
        #     result['result'][prec]={match.group(1):match.group(2), match.group(3):match.group(4)}
        matchs = re.findall(pattern, sout)
        for m in matchs:
            # if not result['result'][prec]:
            #     result['result'][prec] = {}
            result['result'].setdefault(prec, {})
            result['result'][prec]=result['result'][prec] | {m[0]:m[1],m[2]:m[3]}
        pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=1000? \] = ([\d.]+)"
        matchs = re.findall(pattern, sout)
        for m in matchs:
            # if not result['result'][prec]:
            #     result['result'][prec] = {}
            result['result'].setdefault(prec, {})
            result['result'][prec]=result['result'][prec] | {m[0]:m[1]}
        
        
        
        logging.info("**************")
        logging.info(matchs)
        logging.info("**************")

    return result

def run_perf_script(script):
    return run_script(script)

def run_script(script):
    start_time = time.perf_counter()
    result = subprocess.run(script, shell=True, capture_output=True,text=True,executable="/bin/bash")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logging.info(f"执行命令：\n{script}")
    logging.info("执行时间: {:.4f} 秒".format(execution_time))
    print("标准输出:", result.stdout)
    print("标准错误:", result.stderr)
    logging.info("返回码:", result.returncode)
    return result

if __name__ == "__main__":
    main()