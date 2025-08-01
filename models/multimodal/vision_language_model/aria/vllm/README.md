# Aria (vLLM)

## Model Description

Aria is a multimodal native MoE model. It features:

- State-of-the-art performance on various multimodal and language tasks, superior in video and document understanding;
- Long multimodal context window of 64K tokens;
- 3.9B activated parameters per token, enabling fast inference speed and low fine-tuning cost.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | vLLM | Release |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.3.0 | >=0.6.4 | 25.09 |
| MR-V100 | 4.2.0 | >=0.6.6 | 25.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/rhymes-ai/Aria>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "data/Aria"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model data/Aria --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --dtype bfloat16 --tokenizer-mode slow
```

## Model Results
