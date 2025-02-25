# LlaMa3 70B (vLLM)

## Description

This model is the Meta Llama 3 large language model series (LLMs) released by Meta, which is a series of pre-trained and
instruction-tuned generative text models, available in 8B and 70B models. The model is 70B in size and is designed for
large-scale AI applications.

The Llama 3 command-tuned model is optimized for conversational use cases and outperforms many available open source
chat models on common industry benchmarks. In addition, when developing these models, the research team paid great
attention to optimizing beneficialness and safety.

Llama 3 is a major improvement over Llama 2 and other publicly available models:

--Trained on a dataset seven times larger than Llama 2

--Llama 2 has twice the context length of 8K

--Encode the language more efficiently using a larger token vocabulary with 128K tokens

## Setup

### Install

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev
```

### Download

- Model: <https://huggingface.co/Undi95/Meta-Llama-3-70B-hf>

```bash
# Download model from the website and make sure the model's path is "data/Meta-Llama-3-70B-Instruct"
mkdir data

```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 offline_inference.py --model data/Meta-Llama-3-70B-Instruct/ --max-tokens 256 -tp 8 --temperature 0.0
```
