# Xing4.0-29B-A4B (vLLM)

## Model Description

**Xing4.0-29B-A4B** (XingChen-Xing4_0) is a causal language model featuring Multi-head Latent Attention (MLA), a Mixture-of-Experts (MoE) block, and Manifold-constrained Hyper-Connections (mHC) that expand the residual stream into hc_mult parallel streams mixed by input-dependent, doubly-stochastic matrices.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: Xing4.0-29B-A4B 权重（按内部发布渠道下载后放到 `/path/to/Xing4.0-29B-A4B`）
- Plugin: `xing4-vllm-plugin`（`vllm.general_plugins` 入口名 `xing4_0`）

```bash
mkdir -p data/
ln -s /path/to/Xing4.0-29B-A4B ./data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

容器内还需安装 out-of-tree 插件，并将 mHC 的 tilelang 路径改为 torch native：

```bash
pip3 install --no-cache-dir --no-deps /path/to/xing4-vllm-plugin
# CoreX: 将 vllm/.../layers/mhc.py 中 tilelang CUDA 调用改为 self.forward_native
# 并将 HAS_TILELANG_MHC 置为 False
```

## Model Inference

### Inference with offline

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 offline_inference.py \
  --model ./data/Xing4.0-29B-A4B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code \
  --enforce-eager \
  --temperature 0.0 \
  --max-tokens 128
```

### Inference with serve

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve ./data/Xing4.0-29B-A4B \
  --served-model-name Xing4.0-29B-A4B \
  --port 8000 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --trust-remote-code \
  --enforce-eager
```

> CoreX 上 mHC 需走 `forward_native`（torch）路径；`--enforce-eager` 关闭 compile / CUDA Graph。

## Model Results

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Xing4.0-29B-A4B | BF16 | 0.12 | - | 5.23 |

> 以上结果在 eager 模式 + mHC torch native 路径下测得。
