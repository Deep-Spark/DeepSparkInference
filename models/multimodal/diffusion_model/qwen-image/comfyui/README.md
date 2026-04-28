# Qwen-Image (ComfyUI)

## Model Description

 Qwen-Image, an image generation foundation model in the Qwen series that achieves significant advances in complex text rendering and precise image editing. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors>
- Encoder: <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors>
- VAE: <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/vae/qwen_image_vae.safetensors>

### Install Dependencies

```bash
git clone https://github.com/Deep-Spark/ComfyUI
cd ComfyUI
git checkout fix_flux_fp64
pip install -r requirements.txt
python3 main.py --port 8187 --listen 0.0.0.0
```

## Model Inference

- Download qwen_image_fp8_e4m3fn.safetensors and put it in your ComfyUI/models/diffusion_models directory.

- Download qwen_2.5_vl_7b_fp8_scaled.safetensors and put it in your ComfyUI/models/text_encoders directory.

- Download qwen_image_vae.safetensors and put it in your ComfyUI/models/vae/ directory

You can then load up or drag the following image in ComfyUI to get the workflow:

https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/qwen_image_basic_example.png


## References

- [Qwen Image](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/#edit-model)