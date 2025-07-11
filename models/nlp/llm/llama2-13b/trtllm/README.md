# Llama2 13B (TensorRT-LLM)

## Model Description

The Llama2 model is part of the Llama project which aims to unlock the power of large language models. The latest
version of the Llama model is now accessible to individuals, creators, researchers, and businesses of all sizes. It
includes model weights and starting code for pre-trained and fine-tuned Llama language models with parameters ranging
from 7B to 70B.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/meta-llama/Llama-2-13b-chat-hf>
- Dataset:<https://huggingface.co/datasets/cnn_dailymail>

```bash
# Download model from the website and make sure the model's path is "data/llama2-13b-chat"
# Download dataset from the website and make sure the dataset's path is "data/datasets_cnn_dailymail"
mkdir data/

# Please download rouge.py to this path if your server can't attach huggingface.co.
mkdir -p rouge/
wget --no-check-certificate https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/rouge/rouge.py -P rouge
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

bash scripts/set_environment.sh .
```

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

### FP16

```bash
# Build Engine
bash scripts/test_trtllm_llama2_13b_gpu2_build.sh
# Inference
bash scripts/test_trtllm_llama2_13b_gpu2.sh
```

## Model Results

| Model      | tokens | tokens per second |
| ---------- | ------ | ----------------- |
| Llama2 13B | 1596   | 33.39             |
