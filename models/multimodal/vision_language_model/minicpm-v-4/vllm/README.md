# MiniCPM-V-4 (vLLM)

## Model Description

MiniCPM-V 4.0 is the latest efficient model in the MiniCPM-V series. The model is built based on SigLIP2-400M and MiniCPM4-3B with a total of 4.1B parameters. It inherits the strong single-image, multi-image and video understanding performance of MiniCPM-V 2.6 with largely improved efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/openbmb/MiniCPM-V-4>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "./minicpm-v-4"
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
python3 offline_inference_vision_language.py --model-type minicpmv
```

## Model Results

### Benchmarking vLLM

```bash
vllm bench throughput --model ./minicpm-v-4 --backend vllm-chat --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --num-prompts 10  --hf-split train --trust_remote_code
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| MiniCPM-V-4 | BF16 | 1.48 | 631.81 | 189.04 |