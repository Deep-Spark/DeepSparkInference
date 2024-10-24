# StableLm-2-1_6B

## Description

Stable LM 2 1.6B is a decoder-only language model with 1.6 billion parameters. It has been pre-trained on a diverse multilingual and code dataset, comprising 2 trillion tokens, for two epochs. This model is designed for various natural language processing tasks, including text generation and dialogue systems. Due to its extensive training on such a large and diverse dataset, Stable LM 2 1.6B can effectively capture the nuances of language, including grammar, semantics, and contextual relationships, which enhances the quality and accuracy of the generated text.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
pip3 install transformers
```

### Download

-Model: <https://huggingface.co/stabilityai/stablelm-2-1_6b/tree/main>

```bash
# Download model from the website and make sure the model's path is "data/stablelm/stablelm-2-1_6b"
mkdir -p data/stablelm/stablelm-2-1_6b
```

## Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference.py --model ./data/stablelm/stablelm-2-1_6b --max-tokens 256 -tp 1 --temperature 0.0
```

## Results

| Model    | QPS   |
| -------- | ----- |
| StableLM | 254.3 |