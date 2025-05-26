# ViT (IGIE)

## Model Description

ViT is a novel vision model architecture proposed by Google in the paper *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ViT introduces the Transformer architecture, originally designed for natural language processing tasks, into the field of computer vision. By dividing an image into small patches (Patch) and treating them as input tokens, it leverages the self-attention mechanism to perform global feature modeling of the image.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/google/vit-base-patch16-224>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --output vit.onnx

# Use onnxsim optimize onnx model
onnxsim vit.onnx vit_opt.onnx
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

| Model | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
|  ViT  | 32        | FP16      | 432.257 |  81.4    | 95.977   |
