# YOLOv8

## Description

Yolov8 combines speed and accuracy in real-time object detection tasks. With a focus on simplicity and efficiency, this model employs a single neural network to make predictions, enabling fast and accurate identification of objects in images or video streams.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
mkdir -p checkpoints/
mv yolov8n.pt yolov8.pt
python3 export.py --weight yolov8.pt --batch 32
onnxsim yolov8.onnx ./checkpoints/yolov8.onnx
```

## Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov8_int8_accuracy.sh
# Performance
bash scripts/infer_yolov8_int8_performance.sh
```

## Results

| Model  | BatchSize | Precision | FPS      | MAP@0.5 |
| ------ | --------- | --------- | -------- | ------- |
| YOLOv8 | 32        | FP16      | 1511.366 | 0.525   |
| YOLOv8 | 32        | INT8      | 1841.017 | 0.517   |
