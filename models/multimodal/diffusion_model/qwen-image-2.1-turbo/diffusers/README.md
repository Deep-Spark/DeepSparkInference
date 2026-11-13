# Qwen-Image-2.1-Turbo (Diffusers)

## Model Description

**Qwen-Image-2.1-Turbo** is an accelerated checkpoint of Qwen-Image-2.1 for text-to-image generation and image editing. It keeps the same 7B visual generation architecture and runs with a built-in 8-step sampling schedule (CFG=1 by default), with optional prefix KV caching across denoising steps.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo>
- Model: <https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo>

```bash
mkdir -p data/
ln -s /path/to/Qwen-Image-2.1-Turbo ./data/
```

### Install Dependencies

```bash
pip3 install -U huggingface_hub
pip3 install -U --no-deps -r requirements.txt
```

> `--no-deps` 保留镜像自带的 CoreX PyTorch。`huggingface_hub` 需要带依赖安装，diffusers 主干要求比镜像更新的 httpx。

## Model Inference

```bash
python3 inference.py --model ./data/Qwen-Image-2.1-Turbo --size 1024
```

> Turbo 使用 checkpoint 自带的 8-step `sample_sigmas`；单卡可用 `enable_model_cpu_offload`。

## Model Results

### Benchmarking Results

| Model | Precision | Resolution | Steps | Latency |
| :----: | :----: | :----: | :----: | :----: |
| Qwen-Image-2.1-Turbo | BF16 | 1024x1024 | 8 | 41.90 s |

> 以上结果在 `enable_model_cpu_offload` 下测得。

## References

- [diffusers](https://github.com/huggingface/diffusers)
