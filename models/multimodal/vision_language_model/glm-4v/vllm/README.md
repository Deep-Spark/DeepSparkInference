# GLM-4v (vLLM)

## Model Description

GLM-4V-9B is the open-source multimodal version of Zhipu AI's latest generation pre-trained model GLM-4 series. GLM-4V-9B demonstrates exceptional performance in various multimodal evaluations, including bilingual (Chinese and English) multi-turn conversations at a high resolution of 1120 * 1120, comprehensive Chinese-English capabilities, perception reasoning, text recognition, and chart understanding. It surpasses GPT-4-turbo-2024-04-09, Gemini 1.0 Pro, Qwen-VL-Max, and Claude 3 Opus in these aspects.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/THUDM/glm-4v-9b>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model /path/to/glm-4v-9b --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --hf-overrides '{"architectures": ["GLM4VForCausalLM"]}'
```

## Model Results

| Model  | QPS | tokens | Token/s    |
| :----: | :----: | :----: | :----: |
| glm-4v-9b |  0.17  | 239  | 40.795 |