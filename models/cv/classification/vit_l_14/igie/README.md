# ViT_L_14 (IGIE)

## Model Description

The model utilizes the Vision Transformer (ViT-L/14) architecture as its visual encoder, representing a high-performance, large-scale variant in the CLIP family. It partitions input images into fine-grained 14x14 patches, enabling the capture of denser visual details compared to the base model. The architecture consists of 24 Transformer layers with a 1024-dimensional hidden width, eventually mapping features through a linear projection head into a 768-dimensional latent space for robust cross-modal alignment.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |


## Model Preparation

### Prepare Resources

Pretrained model: <https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
pip3 install open_clip_torch
pip3 install timm
```

### Model Conversion

```bash
python3 export.py --model-name ViT-L-14 --weight ViT-L-14.pt --output vit_l_14.onnx

# Use onnxsim optimize onnx model
onnxsim vit_l_14.onnx vit_l_14_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vit_l_14_fp16_accuracy.sh
# Performance
bash scripts/infer_vit_l_14_fp16_performance.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :--------: | :----: | :----: | :----: | :----: | :----: |
|  ViT_L_14  | 32        | FP16      | 135.778 |  71.15   | 92.25   |
