# Llama2_13b_gpu2

## Description
The Llama2 model is part of the Llama project which aims to unlock the power of large language models. The latest version of the Llama model is now accessible to individuals, creators, researchers, and businesses of all sizes. It includes model weights and starting code for pre-trained and fine-tuned Llama language models with parameters ranging from 7B to 70B. 

## Setup

### Install
In order to run the model smoothly, we need the following dependency files:
1. ixrt-xxx.whl
2. tensorrt_llm-xxx.whl
3. ixformer-xxx.whl
Please contact the staff to obtain the relevant installation packages.

```bash
yum install mesa-libGL
bash set_environment.sh
pip3 install Path/To/ixrt-xxx.whl
pip3 install Path/To/tensorrt_llm-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
```

### Download
-Model: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf

-Dataset:https://huggingface.co/datasets/cnn_dailymail

```bash
# Download model from the website and make sure the model's path is "data/llama2-13b-chat"
# Download dataset from the website and make sure the dataset's path is "data/datasets_cnn_dailymail"
mkdir data

# Please download rouge.py to this path if your server can't attach huggingface.co.
mkdir -p rouge/
wget --no-check-certificate https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/rouge/rouge.py -P rouge
```

## Inference
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
