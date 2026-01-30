# InternLM3 (vLLM)

## Model Description

InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning. This model has the following characteristics:

- Enhanced performance at reduced cost: State-of-the-art performance on reasoning and knowledge-intensive tasks surpass models like Llama3.1-8B and Qwen2.5-7B. Remarkably, InternLM3 is trained on only 4 trillion high-quality tokens, saving more than 75% of the training cost compared to other LLMs of similar scale.
- Deep thinking capability: InternLM3 supports both the deep thinking mode for solving complicated reasoning tasks via the long chain-of-thought and the normal response mode for fluent user interactions.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0     |  26.03  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/internlm/internlm3-8b-instruct>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Offline

```bash
python3 offline_inference.py --model internlm/internlm3-8b-instruct --max-tokens 256 --max-model-len 8192 -tp 1 --trust-remote-code --temperature 0.0
```

## Model Results

### Benchmarking vLLM

```bash
vllm bench throughput --model internlm/internlm3-8b-instruct --dataset-name sonnet --dataset-path sonnet.txt --num-prompts 10 --trust-remote-code
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| InternLM3 | BF16 | 1.60 | 1042.09 | 239.82 |