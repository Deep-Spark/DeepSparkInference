# Qwen3.6-27B (vLLM)

## Model Description

Qwen3.6-27B is a multimodal dialogue model of the Qwen3.6 series (architecture `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`). It supports text / image / video input and switchable thinking mode. BF16 native weights, about 52 GB (`model.safetensors.index.json` reports 51.75 GiB / 1199 tensors) / 15 safetensors shards (`model-00001-of-00015` … `model-00015-of-00015`). Hidden size 5120, 64 layers, 24 attention heads / 4 KV heads, FFN intermediate size 17408, vocab size 248320, native context 262144, with vision tower.

Qwen3.6-27B-W8A8 is the INT8 (W8A8) quantized checkpoint of the same architecture (~35 GB / 15 safetensors shards). Structure matches the BF16 original.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | dev-only | 26.09 | — |

> **Note:** 请切换到 release/26.09 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/Qwen/Qwen3.6-27B>
- Model: <https://www.modelscope.cn/models/iluvatar-corex/Qwen3.6-27B-W8A8>

```bash
cd models/nlp/llm/qwen3.6-27b/vllm
mkdir -p data/qwen3
ln -s /path/to/Qwen3.6-27B ./data/qwen3
ln -s /path/to/Qwen3.6-27B-W8A8 ./data/qwen3
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Qwen3.6-27B

#### Inference with offline

```bash
python3 offline_inference.py \
  --model ./data/qwen3/Qwen3.6-27B \
  --max-tokens 256 -tp 4 \
  --trust-remote-code --temperature 0.0 \
  --max-model-len 4096
```

#### Starting Server

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

#### Testing

```bash
curl 127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"Qwen3.6-27B","prompt":"简单介绍一下Qwen3.6模型?","temperature":0.0,"max_tokens":128}'
```

### Qwen3.6-27B-W8A8

#### Starting Server

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,4
python3 -m vllm.entrypoints.openai.api_server \
  --model ./data/qwen3/Qwen3.6-27B-W8A8 \
  --served-model-name Qwen3.6-27B-W8A8 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --port 8000
```

Dense W8A8 INT8 path. Do **not** set `VLLM_W8A8_MOE_USE_W4A8`. A weight reordering message in the startup log confirms the INT8 path is active:

```text
[IluCompressedTensorsW8A8Int8 process_weights_after_loading weight] use NN format
  out_size:5120 in_size:4352 weight:torch.Size([5120, 4352]) -> torch.Size([4352, 5120])
[W8A8GemmInfo] format: NN, k: 5120 -> 5120, n: 3584 -> 3584
```

#### Testing

```bash
curl 127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"Qwen3.6-27B-W8A8","prompt":"简单介绍一下Qwen3.6模型?","temperature":0.0,"max_tokens":128}'
```

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

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,4
vllm bench throughput \
  --model ./data/qwen3/Qwen3.6-27B-W8A8 \
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
| Qwen3.6-27B | BF16 | 0.86 | 596.04 | 129.26 |
| Qwen3.6-27B-W8A8 | INT8 (W8A8) | 1.00 | 688.62 | 149.33 |

Horizontal comparison under the same environment and load:

| Model | Precision | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen3.6-27B | BF16 | 0.86 | 596.04 | 129.26 |
| Qwen3.6-27B-W8A8 | INT8 (W8A8) | 1.00 | 688.62 | 149.33 |
| Qwen3.5-27B | BF16 | 0.84 | 578.67 | 125.49 |

## References

- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
- [vLLM](https://github.com/vllm-project/vllm)
- [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- [DeepSparkInference · Qwen3-8B](https://gitee.com/deep-spark/deepsparkinference/tree/master/models/nlp/llm/qwen3-8b/vllm)
