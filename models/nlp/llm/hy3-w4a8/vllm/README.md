# Hy3-W4A8 (vLLM)

## Model Description

**Hy3** is a 295B-parameter Mixture-of-Experts (MoE) model with 21B active parameters and 3.8B MTP layer parameters, developed by the Tencent Hy Team. Following the Hy3 Preview launch in late April, we gathered feedback from 50+ products and scaled up post-training with higher quality data. Today, we introduce Hy3, which outperforms similar-size models and rivals flagship open-source models with 2-5x parameters. It also shows significant gains in utility across various products and productivity tasks.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/iluvatar-corex/Hy3-W4A8>

```bash
mkdir -p data/
ln -s /path/to/Hy3-W4A8 ./data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

### Inference with offline

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 offline_inference.py \
  --model ./data/Hy3-W4A8 \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --trust-remote-code \
  --enforce-eager \
  --temperature 0.0 \
  --max-tokens 128
```

### Inference with serve

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve ./data/Hy3-W4A8 \
  --served-model-name Hy3-W4A8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --trust-remote-code \
  --reasoning-parser hy_v3 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --enforce-eager
```

## Model Results

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Hy3-W4A8 | W4A8 | 0.14 | 13.13 | 10.24 |

> 以上结果在 eager 模式下测得。
