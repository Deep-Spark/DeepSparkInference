# FLUX.1-Dev (xDiT)

## Model Description

FLUX.1-Dev is a state-of-the-art text-to-image diffusion model developed by Black Forest Labs. It excels at generating high-quality, detailed images from text prompts with exceptional prompt adherence and image quality.

This model runs on the xDiT framework, optimized for Iluvatar CoreX GPUs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/black-forest-labs/FLUX.1-dev>

- Model: <https://modelscope.cn/models/black-forest-labs/FLUX.1-schnell>

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

1. Modify model path in ``run.sh``:
```bash
# The run.sh script is pre-copied in this directory
# Modify MODEL_CONFIGS to point to your model path
vim run.sh
# Update: MODEL_CONFIGS=(["Flux"]="flux_example.py /home/data/flux___1-schnell/ 28")
```

2. Run script:
```bash
bash run.sh
```

3. The model supports 512*512 and 1024*1024 image sizes. To modify:
```bash
vim run.sh
# Modify TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5" 
# to TASK_ARGS="--height 512 --width 512 --no_use_resolution_binning --guidance_scale 3.5"
```

## References

- [FLUX.1](https://github.com/black-forest-labs/flux)
- [xDiT](https://github.com/xdit-team/xDiT)