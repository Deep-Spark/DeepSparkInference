# Qwen3.6-27B (vLLM)

## Model Description

Qwen3.6-27B is a multimodal dialogue model of the Qwen3.6 series (architecture `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`). It supports text / image / video input and switchable thinking mode. BF16 native weights, about 52 GB (`model.safetensors.index.json` reports 51.75 GiB / 1199 tensors) / 15 safetensors shards (`model-00001-of-00015` … `model-00015-of-00015`). Hidden size 5120, 64 layers, 24 attention heads / 4 KV heads, FFN intermediate size 17408, vocab size 248320, native context 262144, with vision tower.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | dev-only | 26.09 | — |

> **Note:** 请切换到 release/26.09 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3.6-27B>

```bash
cd models/nlp/llm/qwen3.6-27b/vllm
mkdir -p data/qwen3
ln -s /path/to/Qwen3.6-27B ./data/qwen3
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with offline

```bash
python3 offline_inference.py \
  --model ./data/qwen3/Qwen3.6-27B \
  --max-tokens 256 -tp 4 \
  --trust-remote-code --temperature 0.0 \
  --max-model-len 4096
```

### Inference with serve

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./data/qwen3/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --port 8000
```

> BF16 dense model. Do **not** set `VLLM_W8A8_MOE_USE_W4A8` (that flag is only for MoE W4A8 checkpoints such as Qwen3.6-35B-A3B-W4A8).

## Model Results

### Benchmarking vLLM

```bash
# If sonnet.txt is missing:
# curl -fsSL -o /tmp/sonnet.txt \
#   https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt

vllm bench throughput \
  --model ./data/qwen3/Qwen3.6-27B \
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
| Qwen3.6-27B | BF16 | — | — | — |

Horizontal comparison under the same environment and load:

| Model | Precision | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen3.6-27B | BF16 | — | — | — |
| Qwen3.6-27B-W8A8 | INT8 (W8A8) | — | — | — |
| Qwen3.5-27B | BF16 | — | — | — |

## References

- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSparkInference · Qwen3-8B](https://gitee.com/deep-spark/deepsparkinference/tree/master/models/nlp/llm/qwen3-8b/vllm)
