# Pixtral (vLLM)

## Model Description

Pixtral is trained to understand both natural images and documents, achieving 52.5% on the MMMU reasoning benchmark, surpassing a number of larger models. The model shows strong abilities in tasks such as chart and figure understanding, document question answering, multimodal reasoning and instruction following. Pixtral is able to ingest images at their natural resolution and aspect ratio, giving the user flexibility on the number of tokens used to process an image. Pixtral is also able to process any number of images in its long context window of 128K tokens. Unlike previous open-source models, Pixtral does not compromise on text benchmark performance to excel in multimodal tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/mistralai/Pixtral-12B-2409>
- Model: <https://modelscope.cn/models/neuralmagic/Pixtral-Large-Instruct-2411-hf-quantized.w4a16>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path are "data/Pixtral-12B-2409" "data/Pixtral-Large-Instruct-2411-hf-quantized.w4a16"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model data/Pixtral-12B-2409 --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --tokenizer-mode 'mistral'

# w4a16
export VLLM_USE_V1=1
python3 offline_inference_2411_w4a16.py --model data/Pixtral-Large-Instruct-2411-hf-quantized.w4a16/
```

## Model Results