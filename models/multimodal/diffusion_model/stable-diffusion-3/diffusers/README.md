# Stable Diffusion 3

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

Download the stable-diffusion-3-medium-diffusers from [huggingface page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).

```bash
mkdir -p stablediffusion3
# download stable-diffusion-3-medium-diffusers into stablediffusion3
```

### Install Dependencies

```bash
pip3 install accelerate scipy safetensors
pip3 uninstall apex
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.35.1-py3-none-any.whl
```

## Model Inference

```bash
python3 demo_sd3.py
```

## References
- [diffusers](https://github.com/huggingface/diffusers)