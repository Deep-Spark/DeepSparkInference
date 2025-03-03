# Qwen2-7B Instruct (vLLM)

## Description

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and
instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This
repo contains the instruction-tuned 7B Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has
generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series
of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics,
reasoning, etc.

Qwen2-7B-Instruct supports a context length of up to 131,072 tokens, enabling the processing of extensive inputs.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# Contact the iluvatar manager to get adapted install packages of vllm, triton, and ixformer
pip3 install vllm
pip3 install triton
pip3 install ixformer
```

### Download

- Model: https://modelscope.cn/models/Qwen/Qwen2-7B-Instruct

```bash
cd models/nlp/large_language_model/qwen2-7b/vllm
mkdir -p data/qwen2
ln -s /path/to/Qwen2-7B-Instruct ./data/qwen2
```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0
python3 offline_inference.py --model ./data/qwen2/Qwen2-7B-Instruct --max-tokens 256 -tp 1 --temperature 0.0
```
