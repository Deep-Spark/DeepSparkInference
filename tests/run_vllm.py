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
    if model["category"] in ["nlp/llm", "multimodal/vision_language_model", "speech/asr"]:
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
        if model["model_name"] == mode_name.lower() and model["framework"] == "vllm":
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
    prepare_script = f"""
    set -x
    cd ../{model['model_path']}
    ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./{model_name}
    pip install /mnt/deepspark/install/xformers-0.0.26.post1+corex.4.3.0-cp310-cp310-linux_x86_64.whl
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
        if model_name == "baichuan2-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./baichuan2-7b/ --max-tokens 256 --trust-remote-code --chat_template template_baichuan.jinja --temperature 0.0
            """
            if prec == "int8":
                script = f"""
                set -x
                cd ../{model['model_path']}
                python3 offline_inference.py --model ./baichuan2-7b/int8/ --chat_template template_baichuan.jinja --quantization w8a16 --max-num-seqs 1 --max-model-len 256 --trust-remote-code --temperature 0.0 --max-tokens 256
                """
        elif model_name == "chatglm3-6b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./chatglm3-6b --trust-remote-code --temperature 0.0 --max-tokens 256
            """
        elif model_name == "chatglm3-6b-32k":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./chatglm3-6b-32k --trust-remote-code --temperature 0.0 --max-tokens 256
            """
        elif model_name == "llama2-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./llama2-7b --max-tokens 256 -tp 1 --temperature 0.0
            """
        elif model_name == "llama3-70b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python3 offline_inference.py --model ./llama3-70b --max-tokens 256 -tp 4 --temperature 0.0
            """
        elif model_name == "qwen-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1
            python3 offline_inference.py --model ./qwen-7b --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
            """
        elif model_name == "qwen1.5-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./qwen1.5-7b --max-tokens 256 -tp 1 --temperature 0.0 --max-model-len 3096
            """
        elif model_name == "qwen1.5-14b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./qwen1.5-14b --max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 896
            """
        elif model_name == "qwen1.5-32b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python3 offline_inference.py --model ./qwen1.5-32b --max-tokens 256 -tp 4 --temperature 0.0
            """
        elif model_name == "qwen1.5-72b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1
            python3 offline_inference.py --model ./qwen1.5-72b --max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 3096
            """
        elif model_name == "qwen2-7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0
            python3 offline_inference.py --model ./qwen2-7b --max-tokens 256 -tp 1 --temperature 0.0
            """
        elif model_name == "qwen2-72b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            python3 offline_inference.py --model ./qwen2-72b --max-tokens 256 -tp 8 --temperature 0.0 --gpu-memory-utilization 0.98 --max-model-len 32768
            """
        elif model_name == "stablelm":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export CUDA_VISIBLE_DEVICES=0,1
            python3 offline_inference.py --model ./stablelm --max-tokens 256 -tp 1 --temperature 0.0
            """
        elif model_name.startswith("deepseek-r1-distill-"):
            if model_name == "deepseek-r1-distill-qwen-32b":
                tp = 4
            else:
                tp = 2
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference.py --model ./{model_name} --max-tokens 256 -tp {tp} --temperature 0.0 --max-model-len 3096
            """
        elif model_name == "aria":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --dtype bfloat16 --tokenizer-mode slow
            """
        elif model_name == "chameleon_7b" or model_name == "fuyu_8b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
            """
        elif model_name == "idefics3":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference_vision_language.py --model-type idefics3
            """
        elif model_name == "h2vol":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --disable-mm-preprocessor-cache
            """
        elif model_name == "minicpm_v":
            script = f"""
            set -x
            cd ../{model['model_path']}
            python3 offline_inference_vision_language.py --model-type minicpmv
            """
        elif model_name == "llama-3.2":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            export VLLM_FORCE_NCCL_COMM=1
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0 --max-model-len 8192 --max-num-seqs 16
            """
        elif model_name == "pixtral":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --tokenizer-mode 'mistral'
            """
        elif model_name == "llava":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --model-type llava-next --max-model-len 4096
            """
        elif model_name == "llava_next_video_7b":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --model-type llava-next-video --modality video --dtype bfloat16
            """
        elif model_name == "intern_vl":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 2048
            """
        elif model_name == "whisper":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_audio_language.py --model ./{model_name} -tp 1 --temperature 0.0 --model-name openai/whisper-large-v3-turbo --max-tokens 200
            """
        elif model_name == "qwen_vl":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            python3 offline_inference_vision_language.py --model ./{model_name} -tp 1 --trust-remote-code --temperature 0.0 --hf-overrides '{"architectures": ["QwenVLForConditionalGeneration"]}'
            """
        elif model_name == "qwen2_vl":
            script = f"""
            set -x
            cd ../{model['model_path']}
            export VLLM_ASSETS_CACHE=../vllm/
            export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
            python3 offline_inference_vision_language.py --model ./{model_name} --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --max-num-seqs 5
            """

        r, t = run_script(script)
        sout = r.stdout
        pattern = r"tokens: (\d+), QPS: ([\d.]+)"
        matchs = re.search(pattern, sout)
        result["result"].setdefault(prec, {"status": "FAIL"})
        logging.debug(f"matchs:\n{matchs}")
        if matchs:
            result["result"][prec]["tokens"] = int(matchs.group(1))
            result["result"][prec]["QPS"] = float(matchs.group(2))
            result["result"][prec]["status"] = "PASS"
        else:
            pattern = r"Maximum concurrency for ([0-9,]+) tokens per request:\s*([0-9.]+)x"
            matchs = re.search(pattern, sout)
            if matchs:
                result["result"][prec]["tokens"] = int(matchs.group(1).replace(',', ''))
                result["result"][prec]["QPS"] = float(matchs.group(2))
                result["result"][prec]["status"] = "PASS"

        result["result"][prec]["Cost time (s)"] = t
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
