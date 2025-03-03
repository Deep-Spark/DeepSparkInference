# YOLOv7

## Description

YOLOv7 is an object detection model based on the YOLO (You Only Look Once) series. It is an improved version of YOLOv5 developed by the Ultralytics team. YOLOv7 aims to enhance the performance and efficiency of object detection through a series of improvements including network architecture, training strategies, and data augmentation techniques, in order to achieve more accurate and faster object detection.

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

Pretrained model: <https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

- 图片目录: Path/To/val2017/*.jpg
- 标注文件目录: Path/To/annotations/instances_val2017.json

### Model Conversion

```bash

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python3 export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 --batch-size 16
mkdir /Your_Projects/To/checkpoints
mv yolov7.onnx /Path/to/checkpoints/yolov7m.onnx
```

## Inference

```bash
export PROJ_DIR=/Path/to/yolov7/ixrt
export DATASETS_DIR=/Path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=/Path/to/yolov7/ixrt
export CONFIG_DIR=config/YOLOV7_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov7_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov7_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov7_int8_accuracy.sh
# Performance
bash scripts/infer_yolov7_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS      |MAP@0.5   |MAP@0.5:0.95 |
--------|-----------|----------|---------|----------|-------------|
YOLOv7  |    32     |   FP16   | 375.41  |  0.693   |  0.506      |
YOLOv7  |    32     |   INT8   | 816.71  |  0.688   |  0.471      |
