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
apt install -y libgl1-mesa-dev

pip3 install tqdm
pip3 install onnx
pip3 install pycocotools
pip3 install ultralytics
```

### Download

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight yolov8n.pt --batch 32
onnxsim yolov8n.onnx ./data/yolov8n.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8n_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8n_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov8n_int8_accuracy.sh
# Performance
bash scripts/infer_yolov8n_int8_performance.sh
```
