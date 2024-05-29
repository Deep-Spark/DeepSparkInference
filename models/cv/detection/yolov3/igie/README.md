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

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install ultralytics
pip3 install pycocotools
```

### Download

Pretrained model: <https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov3.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
python3 export.py --weight yolov3.pt --output yolov3.onnx

# Use onnxsim optimize onnx model
onnxsim yolov3.onnx yolov3_opt.onnx
```

## Inference
```bash
export DATASETS_DIR=/Path/to/coco/
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

Model   |BatchSize  |Precision |FPS      |MAP@0.5  |MAP@0.5:0.95 |
--------|-----------|----------|---------|---------|-------------|
YOLOv3  |    32     |   FP16   | 312.47  |  0.658  |  0.467      |
YOLOv3  |    32     |   INT8   | 711.72  |  0.639  |  0.427      |
