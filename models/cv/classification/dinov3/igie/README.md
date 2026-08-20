# Dinov3 (IGIE)

## Model Description

DINOv3 is a family of versatile vision foundation models that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <hhttps://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, You need to request model download permissions from Meta.

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --model-dir checkpoints --height 224 --width 224 --batch-size 32 --output dinov3_vits16.onnx 

# Use onnxsim optimize onnx model
onnxsim dinov3_vits16.onnx dinov3_vits16_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_dinov3_fp16_accuracy.sh
# Performance
bash scripts/infer_dinov3_fp16_performance.sh
```

## Model Results

| Model            | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Dinov3 | 32        | FP32      | 827.994 | 77.26   | 94.04     |
