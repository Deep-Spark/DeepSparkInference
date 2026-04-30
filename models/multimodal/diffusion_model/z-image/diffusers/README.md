# Z-Image (Diffusers)

## Model Description

Z-Image is an efficient image generation foundation model with single-stream diffusion transformer. Key features:

- **Undistilled Foundation**: As a non-distilled base model, Z-Image preserves the complete training signal
- **Aesthetic Versatility**: Masters a vast spectrum of visual languages-from hyper-realistic photography to anime
- **Enhanced Output Diversity**: Delivers significantly higher variability in composition, facial identity, and lighting
- **Built for Development**: Ideal starting point for LoRA training, ControlNet and semantic conditioning
- **Robust Negative Control**: Responds with high fidelity to negative prompting

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Tongyi-MAI/Z-Image>

## Model Inference

```bash
mkdir -p Tongyi-MAI
# download Z-Image into Tongyi-MAI
python3 demo.py
```
