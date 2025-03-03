# Qwen1.5-32B-Chat (vLLM)

## Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release
the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation,
attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we
have an improved tokenizer adaptive to multiple natural languages and codes. 

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

- Model: <https://modelscope.cn/models/Qwen/Qwen1.5-32B-Chat>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen1.5-32b/vllm
mkdir -p data/qwen1.5
ln -s /path/to/Qwen1.5-32B ./data/qwen1.5
```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 offline_inference.py --model ./data/qwen1.5/Qwen1.5-32B-Chat --max-tokens 256 -tp 4 --temperature 0.0
```
