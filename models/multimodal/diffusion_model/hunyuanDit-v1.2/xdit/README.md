# HunyuanDiT-v1.2-Diffusers (xDiT)

## Model Description

HunyuanDiT-v1.2 is Tencent's advanced text-to-image diffusion model, featuring improved architecture and training for high-quality image generation. It excels at generating detailed, photorealistic images from text descriptions.

This model runs on the xDiT framework, optimized for Iluvatar CoreX GPUs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/dengcao/HunyuanDiT-v1.2-Diffusers>

### Install Dependencies

1. Install Iluvatar CoreX adapted framework:
```bash
pip install diffusers-{version}-py3-none-any.whl
pip install xfuser-{version}+corex.{v.r.m}-py3-none-any.whl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Inference

1. The scripts are pre-copied in this directory. Modify model path:
```bash
vim run_hunyuandit.sh
# Update MODEL_ID to your actual model path
```

2. Run script:
```bash
bash run_hunyuandit.sh
```

3. The model supports BS=1/BS=2. Different BS prompts format:
```bash
# BS1 (default) prompt format
#--prompt "brown dog laying on the ground with a metal bowl in front of him."
# BS2 prompt format
--prompt "brown dog laying on the ground with a metal bowl in front of him." "brown dog laying on the ground with a metal bowl in front of him."
```

## References

- [HunyuanDiT](https://github.com/Tencent/HunyuanDiT)
- [xDiT](https://github.com/xdit-team/xDiT)