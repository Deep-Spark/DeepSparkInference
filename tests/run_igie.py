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

    # 分类模型
    for index, model in enumerate(models):
        avail_models = ["densenet161", "densenet169", "densenet201", "efficientnetv2_rw_t", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_v2", "efficientnet_v2_s", "googlenet", "hrnet_w18", "inception_v3", "mnasnet0_5", "mobilenet_v2", "mobilenet_v3", "mobilenet_v3_large", "mvitv2_base", "regnet_x_1_6gf", "regnet_y_1_6gf", "repvgg", "res2net50", "resnest50", "resnet101", "resnet152", "resnet18", "resnet50", "resnetv1d50", "resnext101_64x4d", "resnext50_32x4d", "se_resnet50", "shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "vgg16", "wide_resnet50"
]
        if model["task_type"] == "cv/classification" and model['name'] in avail_models:
            logging.info(f"{index}, {model['name']}")
            logging.info(json.dumps(model, indent=4))
            d_url = model['download_url']
            test_data = []
            if d_url is not None and (d_url.endswith('.pth') or d_url.endswith('.pt')):
                cur_result = {'name': model['name'], 'result':'', }
                checkpoint_n = d_url.split("/")[-1]

                test_data.append(run_clf_testcase(model))

            logging.info(json.dumps(test_data, indent=4))

        # 检测模型

def run_clf_testcase(model):
    model_name = model['name']
    result = {'name': model_name, 'result':{}, }
    d_url = model['download_url']
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    pip3 install onnx tqdm
    # ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    cd ../{model['relative_path']}
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


def run_perf_script(script):
    return run_script(script)

def run_script(script):
    start_time = time.perf_counter()
    result = subprocess.run(script, shell=True, capture_output=True,text=True,executable="/bin/bash")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logging.info(f"执行命令：\n{script}")
    logging.info("执行时间: {:.4f} 秒".format(execution_time))
    return result

if __name__ == "__main__":
    main()