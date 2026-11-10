# Qwen-Image-2.1 (Diffusers)

## Model Description

We are excited to open-source **Qwen-Image-2.1**, a unified text-to-image generation and image editing model in the Qwen family. With just **7B parameters in its visual generation component** (32 Single-Stream DiT layers), Qwen-Image-2.1 balances generation quality, inference efficiency, and versatility.

For more details, see the [GitHub repo](https://github.com/QwenLM/Qwen-Image-2.1) and [Blog](https://qwen.ai/blog?id=qwen-image-2.1).

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Qwen/Qwen-Image-2.1>

### Install Dependencies

```bash
pip3 install -U huggingface_hub
pip3 install -U --no-deps -r requirements.txt
```

> `--no-deps` 保留镜像自带的 CoreX PyTorch。`huggingface_hub` 需要带依赖安装，diffusers 主干要求比镜像更新的 httpx。

## Model Inference

```bash
mkdir -p data/
ln -s /path/to/Qwen-Image-2.1 ./data/
python3 inference.py --model ./data/Qwen-Image-2.1 --steps 20 --size 1024
```

## Model Results

| Model | Precision | Resolution | Steps | Latency |
| :----: | :----: | :----: | :----: | :----: |
| Qwen-Image-2.1 | BF16 | 1024x1024 | 20 | 104.0 s |

> 以上结果在 `enable_model_cpu_offload` 下测得，不代表全驻显存的性能上限。
> 20 步里第一步耗时 36.6 s，其余各步稳定在 1.2 s 左右——首步包含把各组件依次搬上显卡的开销。

## References

- [diffusers](https://github.com/huggingface/diffusers)
