# Qwen3-Embedding-8B (vLLM)

## Model Description

Qwen3-Embedding-8B is a state-of-the-art text embedding model designed for text embedding and ranking tasks. It achieves exceptional versatility and comprehensive flexibility.

Key features:
- **Exceptional Versatility**: Ranks #1 in MTEB multilingual leaderboard (score 70.58)
- **Comprehensive Flexibility**: Supports custom embedding dimensions (32-4096)
- **Multilingual Capability**: Supports 100+ languages including programming languages
- **Long Context**: 32k context length

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.06`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen3-Embedding-8B>

### Install Dependencies


## Model Inference

```bash
python3 offline_inference.py
```