# Qwen-VL (vLLM)

## Model Description

Qwen-VL (Qwen Large Vision Language Model) is the visual multimodal version of the large model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-VL accepts image, text, and bounding box as inputs, outputs text and bounding box.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen-VL-Chat>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip install matplotlib
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model /path/to/Qwen-VL-Chat -tp 1 --trust-remote-code --temperature 0.0 --hf-overrides '{"architectures": ["QwenVLForConditionalGeneration"]}'
```

## Model Results

| Model  | QPS | tokens | Token/s    |
| :----: | :----: | :----: | :----: |
| Qwen-VL-Chat |  0.901 | 15    | 13.517 |