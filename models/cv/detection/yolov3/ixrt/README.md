# YOLOv3

## Description

YOLOv3 is a influential object detection algorithm.The key innovation of YOLOv3 lies in its ability to efficiently detect and classify objects in real-time with a single pass through the neural network. YOLOv3 divides an input image into a grid and predicts bounding boxes, class probabilities, and objectness scores for each grid cell.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://pjreddie.com/media/files/yolov3.weights>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

- 图片目录: Path/To/val2017/*.jpg
- 标注文件目录: Path/To/annotations/instances_val2017.json

### Model Conversion

```bash

mkdir checkpoints
git clone https://github.com/zldrobit/onnx_tflite_yolov3.git
mv yolov3.weights onnx_tflite_yolov3/weights

# 修改 detect.py 中 torch.onnx.export() 函数的opset_version=11,会在/weights下生成export.onnx
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights

mv export.onnx /Path/to/checkpoints/yolov3.onnx
```

## Inference

```bash
export PROJ_DIR=/Path/to/yolov3/ixrt
export DATASETS_DIR=/Path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=/Path/to/yolov3/ixrt
export CONFIG_DIR=config/YOLOV3_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov3_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov3_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov3_int8_accuracy.sh
# Performance
bash scripts/infer_yolov3_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS      |MAP@0.5   |MAP@0.5:0.95 |
--------|-----------|----------|---------|----------|-------------|
YOLOv3  |    32     |   FP16   | 757.11  |  0.663   |  0.381      |
YOLOv3  |    32     |   INT8   | 1778.34 |  0.659   |  0.356      |
