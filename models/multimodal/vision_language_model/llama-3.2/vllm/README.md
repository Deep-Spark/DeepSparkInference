# Llama-3.2 (vLLM)

## Model Description

The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and
instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only
models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They
outperform many of the available open source and closed chat models on common industry benchmarks.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |
| MR-V100 | 4.2.0 | 25.06 | release/25.06 |

**This model is compatible with IXUCA SDK up to version 4.3.0.**

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/25.12`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/meta-llama/Llama-3.2-1B>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "data/Aria"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
export VLLM_FORCE_NCCL_COMM=1
python3 offline_inference_vision_language.py --model data/LLamaV3.2 --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0 --max-model-len 8192 --max-num-seqs 16
```

## Model Results

### Benchmarking vLLM

```bash
git clone https://github.com/vllm-project/vllm.git -b v0.8.3 --depth=1
python3 vllm/benchmarks/benchmark_throughput.py \
  --model {model_name} \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 10 \
  --hf-split train
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Llama-3.2 | BF16 | 0.26 | 44.36 | 33.34 |