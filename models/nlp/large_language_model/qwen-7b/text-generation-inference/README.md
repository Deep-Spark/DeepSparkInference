# Qwen-7B

## Description

Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

## Setup

### Install

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

```

### Download

-Model: <https://modelscope.cn/models/qwen/Qwen-7B/summary>

```bash
# Make sure the model's file name is qwen-7B
mkdir data
```

## Inference

### Start webserver

#### Single GPU

```bash
# Use one docker container to start webserver
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENABLE_INFER_PG=1 CUDA_VISIBLE_DEVICES=0 USE_FLASH_ATTENTION=true text-generation-launcher --model-id ./data/qwen-7B --sharded false --dtype float16  --disable-custom-kernels --port 8001 --max-input-length 13312 --max-batch-prefill-tokens 13312 --max-total-tokens 15360 --max-batch-total-tokens 15360
```

#### Offline test

```bash
# Use another docker container to run offline test
export CUDA_VISIBLE_DEVICES=1
python3 offline_inference.py --model2path ./data/qwen-7B
```

## Results

| Model   | QPS   |
| ------- | ----- |
| Qwen-7B | 35.64 |
