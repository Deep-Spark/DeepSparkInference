# YOLOv4

## Description

YOLOv4 employs a two-step process, involving regression for bounding box positioning and classification for object categorization. it amalgamates past YOLO family research contributions with novel features like WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, DropBlock regularization, and CIoU loss.

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
pip3 install pycocotools
pip3 install pycuda
```

### Download

Pretrained cfg: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg>
Pretrained model: <https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# clone yolov4
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git yolov4

# download weight
mkdir data
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data

# export onnx model
python3 export.py --cfg yolov4/cfg/yolov4.cfg --weight data/yolov4.weights --batchsize 16 --output data/yolov4.onnx
mv yolov4_16_3_608_608_static.onnx data/yolov4.onnx

# Use onnxsim optimize onnx model
onnxsim data/yolov4.onnx data/yolov4_sim.onnx

# Make sure the dataset path is "data/coco"
```

## Inference

### FP16

```bash
# Accuracy
bash scripts/infer_yolov4darknet_fp16_accuary.sh
# Performance
bash scripts/infer_yolov4darknet_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov4darknet_int8_accuracy.sh
# Performance
bash scripts/infer_yolov4darknet_int8_performance.sh
```

## Results

| Model  | BatchSize | Precision | FPS    | MAP@0.5 |
| ------ | --------- | --------- | ------ | ------- |
| YOLOv4 | 32        | FP16      | 303.27 | 0.730   |
| YOLOv4 | 32        | INT8      | 682.14 | 0.608   |

## Reference

DarkNet: <https://github.com/AlexeyAB/darknet>
Pytorch-YOLOv4: <https://github.com/Tianxiaomo/pytorch-YOLOv4>
