# NVLM (vLLM)

## Model Description

NVLM, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., GPT-4o) and open-access models (e.g., Llama 3-V 405B and InternVL 2). Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 | release/26.09 |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.09`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/nvidia/NVLM-D-72B>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "data/NVLM-D-72B"
mkdir data
```

### Install Dependencies



## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
export VLLM_FORCE_NCCL_COMM=1
python3 offline_inference_vision_language.py --model data/NVLM-D-72B -tp 8

## Model Results