# Stable Diffusion 3

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 |

## Model Preparation

### Prepare Resources

Download the stable-diffusion-3-medium-diffusers from [huggingface page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).

### Install Dependencies

```bash
pip3 install transformers==4.39.3 accelerate==0.29.0 scipy safetensors
pip3 uninstall apex
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/diffusers-0.31.0+corex.4.3.0-py3-none-any.whl
```

## Model Inference

```bash
python3 demo_sd3.py
```

## References

- [diffusers](https://github.com/huggingface/diffusers)
