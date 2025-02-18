# Qwen1.5-7B (vLLM)

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

-Model: <https://modelscope.cn/models/qwen/Qwen1.5-7B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen1.5-7b/vllm
mkdir -p data/qwen1.5
ln -s /path/to/Qwen1.5-7B ./data/qwen1.5
```

## Inference

```bash
python3 offline_inference.py --model ./data/qwen1.5/Qwen1.5-7B --max-tokens 256 -tp 1 --temperature 0.0 --max-model-len 3096
```

## Results

| Model      | QPS   |
| ---------- | ----- |
| Qwen1.5-7B | 109.56|
