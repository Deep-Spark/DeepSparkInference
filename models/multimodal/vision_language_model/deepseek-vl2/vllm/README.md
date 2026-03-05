# DeepSeek-VL2-tiny (vLLM)

## Model Description

Introducing DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL. DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding. Our model series is composed of three variants: DeepSeek-VL2-Tiny, DeepSeek-VL2-Small and DeepSeek-VL2, with 1.0B, 2.8B and 4.5B activated parameters respectively. DeepSeek-VL2 achieves competitive or state-of-the-art performance with similar or fewer activated parameters compared to existing open-source dense and MoE-based models.

More details can be found in https://huggingface.co/deepseek-ai/deepseek-vl2-tiny

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0     |  26.03  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/deepseek-ai/deepseek-vl2-tiny>

```bash
cp -r ../../vllm_public_assets/ ./
mkdir -p deepseek-ai/
# download model into deepseek-ai/deepseek-vl2-tiny
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
python3 offline_inference_vision_language.py --model-type deepseek_vl_v2
```

## Model Results

### Benchmarking vLLM

```bash
vllm bench throughput --model deepseek-ai/deepseek-vl2-tiny --backend vllm-chat --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --num-prompts 10 --hf-split train --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' --max-model-len 4096
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| DeepSeek-VL2-tiny | FP16 | 1.51 | 2209.44 | 193.31 |