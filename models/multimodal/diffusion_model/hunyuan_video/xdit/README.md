# HunyuanVideo (xDiT)

## Model Description

HunyuanVideo is Tencent's advanced text-to-video diffusion model capable of generating high-quality videos from text descriptions. It features excellent motion coherence, visual quality, and text understanding capabilities.

This model runs on the xDiT framework, optimized for Iluvatar CoreX GPUs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Tencent-Hunyuan/HunyuanVideo>

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

1. The scripts are pre-copied in this directory. Modify model path in ``run_hunyuan_video_usp_teacache.sh``:
```bash
vim run_hunyuan_video_usp_teacache.sh
# Update: MODEL_ID="/data/nlp/HunyuanVideo/" to your actual path
```

2. Run script:
```bash
bash run_hunyuan_video_usp_teacache.sh
```

## References

- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [xDiT](https://github.com/xdit-team/xDiT)