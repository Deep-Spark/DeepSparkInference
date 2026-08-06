# YOLOv8n-cls (IGIE)

## Model Description

YOLOv8n-cls is the nano classification variant of Ultralytics YOLOv8. It is pretrained on ImageNet-1k and optimized for efficient image classification with a compact footprint, making it suitable for resource-constrained inference scenarios.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.5.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

```bash
# download the weight from the recommend link
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# Export static-shape ONNX for Relay (batch=32, imgsz=224)
python3 export.py --weight yolov8n-cls.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8n_cls_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8n_cls_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv8n-cls | 32        | FP16      | 29382.22 | 67.31    | 87.32    |
