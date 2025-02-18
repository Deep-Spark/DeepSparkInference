# Qwen1.5-72B (vLLM)

## Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes. For the beta version, temporarily we did not include GQA (except for 32B) and the mixture of SWA and full attention.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

### Download

-Model: <https://modelscope.cn/models/qwen/Qwen1.5-72B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen1.5-72b/vllm
mkdir data/qwen1.5
ln -s /path/to/Qwen1.5-72B ./data/qwen1.5
```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference.py --model ./data/qwen1.5/Qwen1.5-72B --max-tokens 256 -tp 8 --temperature 0.0 --max-model-len 3096
```

## Results

| Model      | QPS   |
| ---------- | ----- |
| Qwen1.5-72B| 41.24 |
