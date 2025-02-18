# LlaMa2 7B (TensorRT-LLM)

## Description

we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

## Setup

### Instal

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev


bash scripts/set_environment.sh .
```

### Download

-Model: <https://huggingface.co/meta-llama/Llama-2-7b-chat>

-Dataset: <https://huggingface.co/datasets/cnn_dailymail>

```bash
# Download model from the website and make sure the model's path is "data/llama2-7b-chat"
# Download dataset from the website and make sure the dataset's path is "data/datasets_cnn_dailymail"
mkdir data

# Please download rouge.py to this path if your server can't attach huggingface.co.
mkdir -p rouge/
wget --no-check-certificate https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/rouge/rouge.py -P rouge
```

## Inference

### FP16

```bash
# Build engine
bash scripts/test_trtllm_llama2_7b_gpu1_build.sh
# Inference
bash scripts/test_trtllm_llama2_7b_gpu1.sh
```
