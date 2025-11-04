# Ultravox (vLLM)

## Model Description

Ultravox is a multimodal model that can consume both speech and text as input (e.g., a text system prompt and voice user message). The input to the model is given as a text prompt with a special <|audio|> pseudo-token, and the model processor will replace this magic token with embeddings derived from the input audio. Using the merged embeddings as input, the model will then generate output text as usual.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b>
- Model: <https://huggingface.co/openai/whisper-large-v3-turbo>
- Model: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>

```bash
mkdir -p meta-llama
# download Llama-3.2-1B-Instruct into meta-llama/Llama-3.2-1B-Instruct
mkdir -p openai
# download whisper-large-v3-turbo into openai/whisper-large-v3-turbo
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip3 install librosa
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_audio_language.py --model /path/to/ultravox-v0_5-llama-3_2-1b
```

## Model Results
