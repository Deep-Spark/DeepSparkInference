# Qwen2-VL (vLLM)

## Model Description

Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc. And can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc. With the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>

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
python3 offline_inference_vision_language.py --model /path/to/Qwen2-VL-7B-Instruct --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --max-num-seqs 5
```

## Model Results