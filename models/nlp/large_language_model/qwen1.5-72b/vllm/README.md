# Qwen1.5-72B

## Description

Qwen1.5 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes. For the beta version, temporarily we did not include GQA (except for 32B) and the mixture of SWA and full attention.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

# Please contact the staff to obtain the relevant installlation packages.
pip3 install Path/To/bitsandbytes-xxx.whl
pip3 install Path/To/flash_atten-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
pip3 install Path/To/vllm-xxx.whl
pip3 install Path/To/eetq-xxx.whl
```

### Download

-Model: <https://modelscope.cn/models/qwen/Qwen1.5-72B/summary>

```bash
mkdir data/qwen1.5
```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference.py --model /data/qwen1.5/$MODEL_ID --max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 3096
```

## Results

| Model      | QPS   |
| ---------- | ----- |
| Qwen1.5-72B| 41.24 |
