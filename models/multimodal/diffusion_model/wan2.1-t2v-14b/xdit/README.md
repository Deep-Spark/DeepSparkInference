# Wan2.1-T2V-14B-Diffusers (xDiT)

## Model Description

Wan2.1-T2V-14B is Wan AI's large-scale text-to-video diffusion model with 14B parameters. It generates high-quality, cinematic videos from text prompts with excellent motion dynamics and visual fidelity.

This model runs on the xDiT framework, optimized for Iluvatar CoreX GPUs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B-Diffusers>

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
vim run_wan_2.1_t2v_14b.sh
# Update MODEL_ID to your actual model path
# Modify TASK_ARGS if needed
```

2. Run script:
```bash
bash run_wan_2.1_t2v_14b.sh
```

3. The model supports BS=1/BS=2. Different BS prompts format:
```bash
# BS1 (default) prompt format
--prompt "一个虎虎生威的老虎" \
--negative_prompt "畸形,光照不好" \
# BS2 prompt format
--prompt "一个虎虎生威的老虎" "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage" \
--negative_prompt "畸形,光照不好" "畸形,光照不好" \
```

## References

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [xDiT](https://github.com/xdit-team/xDiT)