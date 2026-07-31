# Qwen3.5-27B (vLLM)

## Model Description

Qwen3.5-27B is a multimodal dialogue model of the Qwen3.5 series (architecture `Qwen3_5ForConditionalGeneration`). It supports text / image / video input and switchable thinking mode. BF16 native weights, about 52 GB / 11 safetensors shards (`00001-of-00011` … `00011-of-00011`), hidden size 5120, 64 layers, native context 262144.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | dev-only | 26.09 | — |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3.5-27B>

```bash
cd models/nlp/llm/qwen3.5-27b/vllm
mkdir -p data/qwen3
ln -s /path/to/Qwen3.5-27B ./data/qwen3
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with offline

```bash
python3 offline_inference.py \
  --model ./data/qwen3/Qwen3.5-27B \
  --max-tokens 256 -tp 4 \
  --trust-remote-code --temperature 0.0 \
  --max-model-len 4096
```

### Inference with serve

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./data/qwen3/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --port 8000
```

> BF16 dense model. Do **not** set `VLLM_W8A8_MOE_USE_W4A8`.

## Model Results

### Benchmarking vLLM

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,4
# If sonnet.txt is missing:
# curl -fsSL -o /tmp/sonnet.txt \
#   https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt

vllm bench throughput \
  --model ./data/qwen3/Qwen3.5-27B \
  --dataset-name sonnet \
  --dataset-path /tmp/sonnet.txt \
  --num-prompts 10 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### Benchmarking Results

| Model | Precision | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen3.5-27B | BF16 | — | — | — |

## References

- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSparkInference · Qwen3-8B](https://gitee.com/deep-spark/deepsparkinference/tree/master/models/nlp/llm/qwen3-8b/vllm)
- [DeepSparkInference · Qwen2-7B](https://gitee.com/deep-spark/deepsparkinference/tree/master/models/nlp/llm/qwen2-7b/vllm)
