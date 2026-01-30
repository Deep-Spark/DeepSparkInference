# Qwen3-8B (vLLM)

## Model Description

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

- Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
- Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3-8B>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
python3 offline_inference.py --model Qwen/Qwen3-8B --max-tokens 256 -tp 1 --trust-remote-code --temperature 0.0

# You can alse inference with other below Qwen3 model.
python3 offline_inference.py --model Qwen/Qwen3-14B --max-tokens 256 -tp 1 --trust-remote-code --temperature 0.0

python3 offline_inference.py --model Qwen/Qwen3-14B-AWQ --max-tokens 256 -tp 1 --trust-remote-code --temperature 0.0

# Only supports TP1 , TP2 or TP4
python3 offline_inference.py --model Qwen/Qwen3-30B-A3B-AWQ --max-tokens 256 -tp 1 --trust-remote-code --temperature 0.0

python3 offline_inference.py --model Qwen/Qwen3-32B --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0

python3 offline_inference.py --model Qwen/Qwen3-235B-A22B --max-tokens 128 -tp 16  --trust-remote-code --temperature 0.0  --gpu-memory-utilization 0.98 --max-model-len 512
```

## Model Results

### Benchmarking vLLM

```bash
vllm bench throughput --model Qwen/Qwen3-8B --dataset-name sonnet --dataset-path sonnet.txt --num-prompts 10 --trust-remote-code
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen3-8B | BF16 | 1.64 | 1125.52 | 246.46 |
