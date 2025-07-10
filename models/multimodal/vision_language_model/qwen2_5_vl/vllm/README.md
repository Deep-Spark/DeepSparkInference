# Qwen2.5-VL (vLLM)

## Model Description

Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images. Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use. Qwen2.5-VL can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments. Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes. for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip install transformers==4.50.3
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
python3 offline_inference_vision_language.py --model /path/to/Qwen2.5-VL-3B-Instruct/ -tp 4 --trust-remote-code --temperature 0.0 --max-token 256
```

## Model Results
