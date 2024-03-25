# YOLOv8

## Description

Yolov8 combines speed and accuracy in real-time object detection tasks. With a focus on simplicity and efficiency, this model employs a single neural network to make predictions, enabling fast and accurate identification of objects in images or video streams.

## Setup

### Install
```
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install pycocotools
pip3 install ultralytics
```

### Download

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
python3 export.py --weight yolov8s.pt --batch 32
```

## Inference
```bash
export DATASETS_DIR=/Path/to/coco/
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

Model   |BatchSize  |Precision |FPS       |MAP@0.5   |MAP@0.5:0.95 |
--------|-----------|----------|----------|----------|-------------|
yolov8  |    32     |   FP16   | 1002.98  |  0.617   |  0.449      |
yolov8  |    32     |   INT8   | 1392.29  |  0.604   |  0.429      |
