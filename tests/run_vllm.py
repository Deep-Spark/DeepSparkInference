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

import subprocess
import json
import re
import time
import logging
import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple
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

    if model["latest_sdk"] < "4.3.0":
        logging.error(f"model name {model['model_name']} is not support for IXUCA SDK v4.3.0.")
        sys.exit(-1)

    whl_url = os.environ.get("WHL_URL")
    result = {}
    # NLP模型
    if model["category"] in ["nlp/llm", "multimodal/vision_language_model", "speech/asr", "speech/speech_synthesis"]:
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        d_url = model["download_url"]
        if d_url is not None:
            result = run_nlp_testcase(model, whl_url)
            check_model_result(result)
            logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    logging.info(f"Full text result: {result}")

def get_model_config(mode_name):
    with open("model_info.json", mode='r', encoding='utf-8') as file:
        models = json.load(file)

    for model in models['models']:
        if model["model_name"] == mode_name.lower() and (model["framework"] == "vllm" or model["framework"] == "lmdeploy" or model["framework"] == "pytorch"):
            return model
    return

def check_model_result(result):
    status = "PASS"
    for prec in ["fp16", "int8"]:
        if prec in result["result"] and result["result"][prec]["status"] == "FAIL":
            status = "FAIL"
            break
    result["status"] = status

# --- Helper: generate inference script per model ---
# Vision-language model configs
_VISION_MODEL_CONFIGS = {
    "aria": ("offline_inference_vision_language.py", ["--max-model-len 4096", "--max-num-seqs 2", "-tp 4", "--dtype bfloat16", "--disable-mm-preprocessor-cache"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "chameleon_7b": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 2"], None, ["VLLM_ASSETS_CACHE=../vllm/"]),
    "fuyu_8b": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 2"], None, ["VLLM_ASSETS_CACHE=../vllm/"]),
    "idefics3": ("offline_inference_vision_language.py", ["--model-type idefics3"], None, []),
    "h2vol": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--disable-mm-preprocessor-cache"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "minicpm_v": ("offline_inference_vision_language.py", ["--model-type minicpmv"], None, []),
    "llama-3.2": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 2", "--max-model-len 8192", "--max-num-seqs 16"], None, ["VLLM_ASSETS_CACHE=../vllm/", "VLLM_FORCE_NCCL_COMM=1"]),
    "pixtral": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--tokenizer-mode 'mistral'"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "llava": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--model-type llava-next", "--max-model-len 4096"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "llava_next_video_7b": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--model-type llava-next-video", "--modality video", "--dtype bfloat16"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "intern_vl": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 2", "--max-model-len 2048"], None, ["VLLM_ASSETS_CACHE=../vllm/"]),
    "qwen_vl": ("offline_inference_vision_language.py", ["-tp 1", "--hf-overrides '{\"architectures\": [\"QwenVLForConditionalGeneration\"]}'"], None, ["VLLM_ASSETS_CACHE=../vllm/"]),
    "qwen2_vl": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--max-num-seqs 5"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/", "ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1"]),
    "qwen2_5_vl": ("offline_inference_vision_language.py", ["-tp 4", "--max-token 256"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/", "ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1"]),
    "e5-v": ("offline_inference_vision_language_embedding.py", ["--modality \"image\"", "--tensor_parallel_size 1", "--task \"embed\"", "--max_model_len 4096"], None, []),
    "glm-4v": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--hf-overrides '{\"architectures\": [\"GLM4VForCausalLM\"]}'"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "minicpm_o": ("offline_inference_vision_language.py", ["--max-model-len 4096", "--max-num-seqs 2", "--disable-mm-preprocessor-cache"], None, []),
    "phi3_v": ("offline_inference_vision_language.py", ["--max-tokens 256", "-tp 4", "--max-model-len 4096"], "0,1,3,4", ["VLLM_ASSETS_CACHE=../vllm/"]),
    "paligemma": ("offline_inference_vision_language.py", ["--max-tokens 256"], None, ["VLLM_ASSETS_CACHE=../vllm/"]),
}

# Standard LLM configs
_LLM_CONFIGS = {
    "chatglm3-6b": ("--trust-remote-code --temperature 0.0 --max-tokens 256", None),
    "chatglm3-6b-32k": ("--trust-remote-code --temperature 0.0 --max-tokens 256", None),
    "llama2-7b": ("--max-tokens 256 -tp 1 --temperature 0.0", None),
    "qwen-7b": ("--max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0", None),
    "qwen1.5-7b": ("--max-tokens 256 -tp 1 --temperature 0.0 --max-model-len 3096", None),
    "qwen1.5-14b": ("--max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 896", None),
    "qwen1.5-32b": ("--max-tokens 256 -tp 4 --temperature 0.0", "0,1,3,4"),
    "qwen2-7b": ("--max-tokens 256 -tp 1 --temperature 0.0", "0"),
    "stablelm": ("--max-tokens 256 -tp 1 --temperature 0.0", "0,1"),
}

def _build_inference_script(model: Dict[str, Any], prec: str) -> str:
    model_name = model["model_name"]
    model_path = model["model_path"]
    checkpoint_n = model["download_url"].split("/")[-1]
    base_script = f"set -x\ncd ../{model_path}\n"

    if model_name in [
        "deepseek-r1-distill-llama-70b","llama3-70b",
        "qwen1.5-72b"]:
        return base_script + f"python3 offline_inference.py --model ./{model_name} --max-tokens 256 -tp 8 --temperature 0.0 --max-model-len 3096"
    elif model_name == "qwen2-72b":
        return base_script + f"python3 offline_inference.py --model ./{model_name} --max-tokens 256 -tp 8 --temperature 0.0 --gpu-memory-utilization 0.98 --max-model-len 32768"
    elif model_name == "nvlm":
        return base_script + f"""
            export VLLM_ASSETS_CACHE=../vllm/
            export VLLM_FORCE_NCCL_COMM=1
            python3 offline_inference_vision_language.py --model ./{model_name} -tp 8
            """

    # Handle models with prefix (cannot use match)
    if model_name.startswith("deepseek-r1-distill-"):
        tp = "4" if model_name == "deepseek-r1-distill-qwen-32b" else "2"
        gpus = "0,1,3,4" if tp == "4" else None
        gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
        return base_script + f"{gpu_prefix}python3 offline_inference.py --model ./{model_name} --max-tokens 256 -tp {tp} --temperature 0.0 --max-model-len 3096"

    # Use match-case for exact model names
    match model_name:
        case "baichuan2-7b":
            if prec == "int8":
                cmd = "python3 offline_inference.py --model ./baichuan2-7b/int8/ --chat_template template_baichuan.jinja --quantization w8a16 --max-num-seqs 1 --max-model-len 256 --trust-remote-code --temperature 0.0 --max-tokens 256"
            else:
                cmd = "python3 offline_inference.py --model ./baichuan2-7b/ --max-tokens 256 --trust-remote-code --chat_template template_baichuan.jinja --temperature 0.0"
            return base_script + cmd

        case "internlm3":
            return base_script + f"python3 offline_inference.py --model-path /mnt/deepspark/data/checkpoints/{checkpoint_n} --tp 1"

        case "cosyvoice":
            return base_script + "cd CosyVoice\npython3 inference_test.py"

        case "xlmroberta":
            return base_script + (
                "python3 offline_inference_scoring.py --model ./xlmroberta --task \"score\" --tensor-parallel-size 1\n"
                "ln -s /mnt/deepspark/data/checkpoints/multilingual-e5-large ./\n"
                "python3 offline_inference_embedding.py --model ./multilingual-e5-large -tp 2"
            )

        case "whisper":
            return base_script + (
                "export VLLM_ASSETS_CACHE=../vllm/\n"
                "python3 offline_inference_audio_language.py --model ./whisper -tp 1 --temperature 0.0 "
                "--model-name openai/whisper-large-v3-turbo --max-tokens 200"
            )

        # Vision-language models
        case "aria" | "chameleon_7b" | "fuyu_8b" | "idefics3" | "h2vol" | "minicpm_v" | "llama-3.2" | "pixtral" | "llava" | "llava_next_video_7b" | "intern_vl" | "qwen_vl" | "qwen2_vl" | "qwen2_5_vl" | "e5-v" | "glm-4v" | "minicpm_o" | "phi3_v" | "paligemma":
            config = _VISION_MODEL_CONFIGS[model_name]
            script_file, args, gpus, envs = config
            env_lines = "\n".join(f"export {e}" for e in envs) + ("\n" if envs else "")
            gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
            arg_str = " ".join(args)
            cmd = f"{gpu_prefix}python3 {script_file} --model ./{model_name} {arg_str} --trust-remote-code --temperature 0.0"
            return base_script + env_lines + cmd

        # Standard LLMs
        case "chatglm3-6b" | "chatglm3-6b-32k" | "llama2-7b" | "qwen-7b" | "qwen1.5-7b" | "qwen1.5-14b" | "qwen1.5-32b" | "qwen2-7b" | "stablelm":
            args, gpus = _LLM_CONFIGS[model_name]
            gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpus} " if gpus else ""
            return base_script + f"{gpu_prefix}python3 offline_inference.py --model ./{model_name} {args}"

        case _:
            # Fallback
            return base_script + f"python3 offline_inference.py --model ./{model_name} --temperature 0.0"

# --- Helper: append benchmark script if needed ---
def _append_benchmark_script(script: str, model: Dict[str, Any]) -> str:
    model_name = model["model_name"]
    category = model["category"]

    excluded_llms = {"baichuan2-7b", "llama2-7b", "qwen-7b", "stablelm", "deepseek-r1-distill-llama-70b","llama3-70b",
        "qwen1.5-72b", "qwen2-72b"}
    excluded_vlms = {
        "fuyu_8b", "chameleon_7b", "llava", "llava_next_video_7b", "paligemma",
        "glm-4v", "qwen_vl", "pixtral", "xlmroberta", "nvlm"
    }

    common_bench = (
        "\npip3 install datasets\n"
        "cp -r /mnt/deepspark/data/repos/vllm ./\n"
    )

    if category == "nlp/llm" and model_name not in excluded_llms:
        if model_name == "qwen1.5-14b":
            bench = (
                "python3 vllm/benchmarks/benchmark_throughput.py --model ./qwen1.5-14b "
                "--dataset-name sonnet --dataset-path vllm/benchmarks/sonnet.txt "
                "--num-prompts 10 --trust_remote_code --max-model-len 896 -tp 2"
            )
        else:
            bench = (
                "CUDA_VISIBLE_DEVICES=0,1,3,4 python3 vllm/benchmarks/benchmark_throughput.py "
                f"--model ./{model_name} --dataset-name sonnet --dataset-path vllm/benchmarks/sonnet.txt "
                "--num-prompts 10 --trust_remote_code --max-model-len 3096 -tp 4"
            )
        return script + common_bench + bench

    if category == "multimodal/vision_language_model" and model_name not in excluded_vlms:
        bench = (
            "mkdir -p lmarena-ai\n"
            "ln -s /mnt/deepspark/data/datasets/VisionArena-Chat lmarena-ai/\n"
            "CUDA_VISIBLE_DEVICES=0,1,3,4 python3 vllm/benchmarks/benchmark_throughput.py "
            f"--model ./{model_name} --backend vllm-chat --dataset-name hf "
            "--dataset-path lmarena-ai/VisionArena-Chat --num-prompts 10 --hf-split train "
            "-tp 4 --max-model-len 4096 --max-num-seqs 2 --trust_remote_code"
        )
        return script + common_bench + bench

    return script


# --- Helper: parse script output into result dict ---
def _parse_script_output(sout: str, prec: str, display_name: str) -> Dict[str, Any]:
    result_entry = {"status": "FAIL"}

    # Primary pattern
    pattern = r"requests: (\d+), QPS: ([\d.]+), tokens: ([\d.]+), Token/s: ([\d.]+)"
    match = re.search(pattern, sout)
    if match:
        result_entry.update({
            "requests": int(match.group(1)),
            "QPS": float(match.group(2)),
            "tokens": int(float(match.group(3))),
            "Token/s": float(match.group(4)),
            "status": "PASS"
        })

        # Benchmark metrics
        benchmark_match = re.search(r"Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s, ([\d.]+) output tokens/s", sout)
        precision_match = re.search(r"is likely:\s*([A-Z0-9]+(?:\s*\([^)]*\))?|OTHER\s*\([^)]*\))", sout)
        precision_value = "UNKNOWN"
        if precision_match:
            prec_str = precision_match.group(1).strip()
            main_match = re.match(r'^([A-Z0-9]+)', prec_str)
            precision_value = main_match.group(1) if main_match else ("OTHER" if prec_str.startswith("OTHER") else prec_str)

        if benchmark_match:
            result_entry.update({
                "Benchmark QPS": float(benchmark_match.group(1)),
                "Benchmark Total TPS": float(benchmark_match.group(2)),
                "Benchmark Output TPS": float(benchmark_match.group(3)),
                "Benchmark Markdown": f"| {display_name} | {precision_value} | {benchmark_match.group(1)} | {benchmark_match.group(2)} | {benchmark_match.group(3)} |"
            })
        else:
            result_entry["Benchmark Markdown"] = f"| {display_name} | {precision_value} | N/A | N/A | N/A |"

        return result_entry

    # Fallback pattern for concurrency
    match = re.search(r"Maximum concurrency for ([0-9,]+) tokens per request:\s*([0-9.]+)x", sout)
    if match:
        return {
            "tokens": int(match.group(1).replace(",", "")),
            "QPS": float(match.group(2)),
            "status": "PASS"
        }

    # Final fallback: generic success message
    if "Offline inference is successful!" in sout:
        return {"status": "PASS"}

    return result_entry


# --- Main function (now simple and low complexity) ---
def run_nlp_testcase(model: Dict[str, Any], whl_url: str) -> Dict[str, Any]:
    get_num_devices_script = "ixsmi -L | wc -l"
    result, _ = run_script(get_num_devices_script)
    num_devices = int(result.stdout.strip())
    logging.info(f"Detected number of GPU devices: {num_devices}")
    model_name = model["model_name"]
    checkpoint_n = model["download_url"].split("/")[-1]
    if num_devices < 8 and model_name in [
        "deepseek-r1-distill-llama-70b","llama3-70b",
        "qwen1.5-72b","qwen2-72b","nvlm"]:
        logging.warning(f"Skipping test for {model_name} due to insufficient GPU devices ({num_devices} detected).")
        return {"name": model_name, "result": {}, "status": "SKIPPED"}
    
    if model_name in ["qwen3-235b","step3","ultravox"]:
        logging.warning(f"Skipping test for {model_name} as it is not supported in this script.")
        return {"name": model_name, "result": {}, "status": "SKIPPED"}

    prepare_script = f"""
set -x
cd ../{model['model_path']}
ln -s /mnt/deepspark/data/checkpoints/{checkpoint_n} ./{model_name}
pip install {whl_url}`curl -s {whl_url} | grep -o 'xformers-[^"]*\.whl' | head -n1`
bash ci/prepare.sh
"""

    if utils.is_debug():
        pip_list = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list + prepare_script + pip_list

    run_script(prepare_script)

    result = {"name": model_name, "result": {}}

    for prec in model["precisions"]:
        logging.info(f"Start running {model_name} {prec} test case")
        script = _build_inference_script(model, prec)
        script = _append_benchmark_script(script, model)
        r, t = run_script(script)
        sout = r.stdout

        parsed = _parse_script_output(sout, prec, model["display_name"])
        parsed["Cost time (s)"] = t
        result["result"][prec] = parsed

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
