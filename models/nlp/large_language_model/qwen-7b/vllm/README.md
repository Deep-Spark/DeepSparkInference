# Qwen-7B 

## Description
Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

## Setup

### Install
In order to run the model smoothly, we need the following dependency files:
1. ixrt-xxx.whl
2. ixformer-xxx.whl
3. vllm-xxx.whl
Please contact the staff to obtain the relevant installation packages.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install transformers==4.33.2
pip3 install Path/To/ixrt-xxx.whl
pip3 install Path/To/vllm-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
```

### Download
-Model: https://modelscope.cn/models/qwen/Qwen-7B/summary

```bash
# Make sure the model's file name is qwen-7B
mkdir data/
ls data/qwen-7B
```

## Run model

```bash
python3 offline_inference.py --model ./data/qwen-7B --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
```

