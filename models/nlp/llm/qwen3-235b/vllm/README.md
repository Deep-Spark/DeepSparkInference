# Qwen3_Moe (vLLM)

## Model Description

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

- Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
- Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only | 25.12 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B>
- Model: <https://www.modelscope.cn/models/swift/Qwen3-235B-A22B-Instruct-2507-AWQ>

```bash
# BF16到W4A8量化模型
python3 bf16ToInt4.py --input-fp8-hf-path /path/to/Qwen3-235B-A22B --output-int8-hf-path ./Qwen3-235B-A22B-w4a8-TN --group-size -1 --format TN --version 2 
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Qwen3-235B-A22B-Instruct
#### Starting Server
```bash
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
--model /path/to/Qwen3-235B-A22B-Instruct-2507-AWQ/ \
--gpu-memory-utilization 0.92 --port 12347 \
--trust-remote-code \
--disable-cascade-attn \
--max-model-len 262144 --seed 42 -tp 4 -pp 4 -dp 1 --max-num-seqs 8
```

#### Testing
```bash
curl 127.0.0.1:12347/v1/completions -H "Content-Type: application/json" -d '{"model":"/path/to/Qwen3-235B-A22B-Instruct-2507-AWQ/",
"prompt":"简单介绍一下Qwen模型?",
"temperature":0.0,
"max_tokens":128}'
```

### Qwen3-235B-A22B-W4A8
#### Starting Server
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" VLLM_W8A8_MOE_USE_W4A8=1 VLLM_USE_V1=1 \ python3 -m vllm.entrypoints.openai.api_server --model ./Qwen3-235B-A22B-w4a8-TN \
--gpu-memory-utilization 0.92 --port 12347 \
--trust-remote-code \
--disable-cascade-attn \ 
--seed 42 -tp 4 -pp 4 -dp 1 --max-num-seqs 8
```

#### Testing
```bash
curl 127.0.0.1:12347/v1/completions -H "Content-Type: application/json" -d '{"model":"./Qwen3-235B-A22B-w4a8-TN", "prompt":"简单介绍一下Qwen3模型?", "temperature":0.0, "max_tokens":128}'
```

## Model Results