# YOLOv11

## Description

YOLOv11 is the latest generation of the YOLO (You Only Look Once) series object detection model released by Ultralytics. Building upon the advancements of previous YOLO models, such as YOLOv5 and YOLOv8, YOLOv11 introduces comprehensive upgrades to further enhance performance, flexibility, and usability. It is a versatile deep learning model designed for multi-task applications, supporting object detection, instance segmentation, image classification, keypoint pose estimation, and rotated object detection.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>

## Model Conversion

```bash
python3 export.py --weight yolo11n.pt --batch 32
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov11_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov11_fp16_performance.sh
```

## Results

| Model   | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------- | ------- | ------------ |
| YOLOv11 | 32        | FP16      | 1519.25 | 0.551   | 0.393        |

## Reference

YOLOv11: <https://docs.ultralytics.com/zh/models/yolo11/>
