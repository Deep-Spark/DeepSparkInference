# Fuyu-8B

## Description

Fuyu-8B is a multi-modal text and image transformer trained by Adept AI.

Architecturally, Fuyu is a vanilla decoder-only transformer - there is no image encoder. Image patches are instead linearly projected into the first layer of the transformer, bypassing the embedding lookup. We simply treat the transformer decoder like an image transformer (albeit with no pooling and causal attention).

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

- Model: <https://huggingface.co/adept/fuyu-8b>

```bash
# Download model from the website and make sure the model's path is "data/fuyu-8b"
mkdir data
```

## Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model ./data/fuyu-8b --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
```
