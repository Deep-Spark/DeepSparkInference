# MiniCPM-V-2 (vLLM)

## Model Description

MiniCPM-V 2 is a compact and efficient language model designed for various natural language processing (NLP) tasks.
Building on its predecessor, MiniCPM-V-1, this model integrates advancements in architecture and optimization
techniques, making it suitable for deployment in resource-constrained environments.s

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/openbmb/MiniCPM-V-2_6>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "./minicpm-v-2"
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip install timm==0.9.10
```

## Model Inference

```bash
python3 offline_inference_vision_language.py --model-type minicpmv
```

## Model Results

### Benchmarking vLLM

```bash
git clone https://github.com/vllm-project/vllm.git -b v0.8.3 --depth=1
python3 vllm/benchmarks/benchmark_throughput.py \
  --model ./minicpm-v-2 \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 10 \
  --hf-split train
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| MiniCPM-V-2 | BF16 | 0.59 | 251.25 | 75.32 |