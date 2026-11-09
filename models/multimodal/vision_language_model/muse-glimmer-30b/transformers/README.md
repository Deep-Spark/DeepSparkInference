# Muse-Glimmer-30B (Transformers)

## Model Description

Muse Glimmer is a 30-billion-parameter causal language model with a dedicated perception encoder, distilled from Muse Spark and purpose-built for autonomous agentic tasks on consumer hardware. The model integrates multi-step reasoning, reliable tool use, multimodal understanding, and failure recovery into a single model that runs locally without requiring cloud infrastructure or network access.

Building effective agents requires key capabilities working together to achieve the user’s goals: instruction following, multimodal understanding, long-context reasoning, tool calling, and recovery from failed steps.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/meta-models/Muse-Glimmer-30B>
- Image: any image file; the shared assets under `../../vllm_public_assets/` can be used.

### Install Dependencies

```bash
pip3 install --no-deps accelerate
```

> 镜像自带的 transformers 已支持 `muse_glimmer`，不要降级；`accelerate` 只用于 `device_map="auto"` 的多卡切分。
> BF16 权重约 55.5 GiB，建议用 3 张及以上 BI-V150；卡不够时放不下的模块会落到内存，推理耗时随之变长。

## Model Inference

```bash
cp -r ../../vllm_public_assets/ ./
mkdir -p data/
ln -s /path/to/Muse-Glimmer-30B ./data/
python3 offline_inference.py \
  --model ./data/Muse-Glimmer-30B \
  --image ./vllm_public_assets/cherry_blossom.jpg \
  --prompt "Describe this image in one sentence."
```

## Model Results

| Model | Precision | Check | Prompt Tokens | Output Tokens | Load | Latency | Result |
| :----: | :----: | :---- | :----: | :----: | :----: | :----: | :----: |
| Muse-Glimmer-30B | BF16 | Describe image in one sentence | 2711 | 87 | 71.5 s | 38.0 s | Pass |

> 该结果用上面的命令在 2 × BI-V150 上测得，输入 `vllm_public_assets/cherry_blossom.jpg`，
> 输出为 `The Tokyo Skytree rises through a canopy of pink cherry blossoms framed against a vivid blue sky.`
>
> 权重放不下时有 4 个模块落到内存，耗时包含这部分搬运，不代表全驻显存的性能。
> 另外该模型按多通道格式生成，解码出的原文里会先出现 `to=self` 等推理通道内容，
> 最终答案在最后一段；脚本按原样打印，不做通道裁剪。

## References

- [transformers](https://github.com/huggingface/transformers)
