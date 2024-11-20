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

    for index, model in enumerate(models):
        if model["task_type"] == "cv/classification" and model['name']=="alexnet":
            print(f"{index}, {model['name']}")
            d_url = model['download_url']
            test_data = []
            if d_url is not None and (d_url.endswith('.pth') or d_url.endswith('.pt')):
                cur_result = {'name': model['name'], 'result':'', }
                checkpoint_n = d_url.split("/")[-1]
                # 执行后获取结果
                script = f"""
                pwd
                pip3 install onnx tqdm
                # ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
                python3 export.py --weight /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} --output alexnet.onnx

                export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/imagenet-val
                cd ../{model['relative_path']}

                # echo "****run fp16 acc ****"
                # bash scripts/infer_alexnet_fp16_accuracy.sh
                # echo "****run fp16 perf ****"
                bash scripts/infer_alexnet_fp16_performance.sh

                """

                # result = subprocess.run(script, shell=True, capture_output=True,text=True,executable="/bin/bash")
                # print("标准输出:", result.stdout)
                # print("标准错误:", result.stderr)
                # print("返回码:", result.returncode)
                # if result.returncode ==0:
                #     test_data.append(cur_result)
                test_data.append(run_clf_testcase(model))
                
            print(json.dumps(test_data, indent=4))
        
def run_clf_testcase(model):
    result = {'name': model['name'], 'result':{}, }
    d_url = model['download_url']
    checkpoint_n = d_url.split("/")[-1]
    prepare_script = f"""
    pip3 install onnx tqdm
    # ln -s /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} ./
    python3 export.py --weight /mnt/deepspark/data/checkpoints/igie/{checkpoint_n} --output alexnet.onnx
    """
    run_script(prepare_script)

    # for prec in model['precisions']:
    for prec in ['fp16']:
        print(f"run {prec} test case")
        script = f"""
        export DATASETS_DIR=/mnt/deepspark/volumes/mdb/data/datasets/imagenet-val
        cd ../{model['relative_path']}
        bash scripts/infer_alexnet_{prec}_accuracy.sh
        bash scripts/infer_alexnet_{prec}_performance.sh
        """
        
        r = run_script(script)
        sout = r.stdout
        print("标准输出:", r.stdout)
        print("标准错误:", r.stderr)
        print("返回码:", r.returncode)
        pattern = r"\* ([\w\d ]+):\s*([\d.]+ [ms%]+)\s*, ([\w\d ]+):\s*([\d.]+ [ms%]+)"
        # match = re.search(pattern, sout)
        # if match:
        #     result['result'][prec]={match.group(1):match.group(2), match.group(3):match.group(4)}
        matchs = re.findall(pattern, sout)
        for m in matchs:
            result['result'][prec]={m[0]:m[1],m[2]:m[3]}

        print(matchs)

    return result
    

def run_perf_script(script):
    return run_script(script)

def run_script(script):
    start_time = time.perf_counter()
    result = subprocess.run(script, shell=True, capture_output=True,text=True,executable="/bin/bash")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("执行时间: {:.4f} 秒".format(execution_time))
    return result
    
if __name__ == "__main__":
    main()