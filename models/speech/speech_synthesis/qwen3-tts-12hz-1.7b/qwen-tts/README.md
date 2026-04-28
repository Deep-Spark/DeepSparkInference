# Qwen3-TTS-12Hz-1.7B-Base (Qwen-TTS)

## Model Description

Qwen3-TTS is a powerful text-to-speech model covering 10 major languages and multiple dialectal voice profiles. The Base model is capable of 3-second rapid voice clone from user audio input.

Key features:
- **Powerful Speech Representation**: Powered by Qwen3-TTS-Tokenizer-12Hz for efficient acoustic compression
- **Universal End-to-End Architecture**: Realizes full-information end-to-end speech modeling
- **Extreme Low-Latency Streaming Generation**: End-to-end synthesis latency as low as 97ms
- **Intelligent Text Understanding**: Supports speech generation driven by natural language instructions

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>
- Reference Audio: <https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav>

### Install Dependencies

```bash
pip install -U qwen-tts
```

## Model Inference

```bash
python3 inference.py
```