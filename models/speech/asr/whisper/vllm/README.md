# Whisper (vLLM)

## Model Description

Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford et al. from OpenAI. Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many datasets and domains in a zero-shot setting.

Whisper large-v3-turbo is a finetuned version of a pruned Whisper large-v3. In other words, it's the exact same model, except that the number of decoding layers have reduced from 32 to 4. As a result, the model is way faster, at the expense of a minor quality degradation. You can find more details about it in this GitHub discussion.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/openai/whisper-large-v3-turbo>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip3 install librosa
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_audio_language.py --model /path/to/whisper-large-v3-turbo/ -tp 1 --temperature 0.0 --model-name openai/whisper-large-v3-turbo --max-tokens 200
```

## Model Results

| Model  | QPS | tokens | Token/s    |
| :----: | :----: | :----: | :----: |
| whisper-large-v3-turbo | 9.186 | 1300    | 597.146 |