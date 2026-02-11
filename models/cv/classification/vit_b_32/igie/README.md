# ViT_B_32 (IGIE)

## Model Description

The model utilizes the Vision Transformer (ViT-B/32) architecture as its visual encoder. It partitions input images into 32x32 patches, which are processed through a feature extractor consisting of 12 Transformer layers. Finally, a linear projection head maps the features into a 512-dimensional latent space, achieving cross-modal alignment between image features and text semantics.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |


## Model Preparation

### Prepare Resources

Pretrained model: <https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
pip3 install open_clip_torch
pip3 install timm
```

### Model Conversion

```bash
python3 export.py --model-name ViT-B-32 --weight ViT-B-32.pt --output vit_b_32.onnx

# Use onnxsim optimize onnx model
onnxsim vit_b_32.onnx vit_b_32_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_vit_fp16_accuracy.sh
# Performance
bash scripts/infer_vit_fp16_performance.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :--------: | :----: | :----: | :----: | :----: | :----: |
|  ViT_B_32  | 32        | FP16      | 3303.136 |  58.16   | 85.337   |
