# Qwen3-ASR-1.7B (Qwen-ASR)

## Model Description

Qwen3-ASR is a state-of-the-art automatic speech recognition model supporting 52 languages and 22 Chinese dialects. The 1.7B version achieves excellent performance while maintaining high inference speed.

Key features:
- **All-in-one**: Supports language identification and ASR for 30 languages and 22 Chinese dialects
- **Excellent and Fast**: Achieves strong recognition under complex acoustic environments
- **Novel Forced Alignment**: Supports timestamp prediction for up to 5 minutes of speech
- **Comprehensive Inference Toolkit**: Supports vLLM-based batch inference, streaming inference, timestamp prediction

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Test Audio (English): <https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav>
- Test Audio (Chinese): <https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav>

### Install Dependencies

```bash
pip install -U qwen-asr
```

## Model Inference

```bash
python3 offline_inference.py
```
