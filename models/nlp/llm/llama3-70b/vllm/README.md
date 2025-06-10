# Llama3 70B (vLLM)

## Model Description

Llama 3 is Meta's latest large language model series, representing a significant advancement in open-source AI
technology. Available in 8B and 70B parameter versions, it's trained on a dataset seven times larger than its
predecessor, Llama 2. The model features an expanded 8K context window and a 128K token vocabulary for more efficient
language encoding. Optimized for conversational AI, Llama 3 demonstrates superior performance across various industry
benchmarks while maintaining strong safety and beneficialness standards. Its 70B version is particularly designed for
large-scale AI applications, offering enhanced reasoning and instruction-following capabilities.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Undi95/Meta-Llama-3-70B-hf>

```bash
# Download model from the website and make sure the model's path is "data/Meta-Llama-3-70B-Instruct"
mkdir data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 offline_inference.py --model data/Meta-Llama-3-70B-Instruct/ --max-tokens 256 -tp 8 --temperature 0.0
```
