# Qwen3-Embedding-8B (vLLM)

## Model Description

Qwen3-Embedding-8B is a state-of-the-art text embedding model designed for text embedding and ranking tasks. It achieves exceptional versatility and comprehensive flexibility.

Key features:
- **Exceptional Versatility**: Ranks #1 in MTEB multilingual leaderboard (score 70.58)
- **Comprehensive Flexibility**: Supports custom embedding dimensions (32-4096)
- **Multilingual Capability**: Supports 100+ languages including programming languages
- **Long Context**: 32k context length

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen3-Embedding-8B>

### Install Dependencies


## Model Inference

```bash
python3 offline_inference.py
```