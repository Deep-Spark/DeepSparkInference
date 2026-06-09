# Wan2.2-TI2V-5B-Diffusers (xDiT)

## Model Description

Wan2.2-TI2V-5B is Wan AI's image-to-video diffusion model with 5B parameters. It generates smooth, high-quality videos from input images, maintaining visual consistency and adding natural motion.

This model runs on the xDiT framework, optimized for Iluvatar CoreX GPUs.

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

- Model: <https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers>

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
vim run_wan_2.2_t2v_5b.sh
# Update MODEL_ID to your actual model path
# Modify TASK_ARGS if needed
```

2. Run script:
```bash
bash run_wan_2.2_t2v_5b.sh
```

## References

- [Wan2.2](https://github.com/Wan-Video/Wan2.1)
- [xDiT](https://github.com/xdit-team/xDiT)