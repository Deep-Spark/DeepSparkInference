# YOLOv7

## Description

YOLOv7 is a state-of-the-art real-time object detector that surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS.

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

Pretrained model: <https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion

```bash
# clone yolov7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

# export onnx model
python3 export.py --weights ../yolov7.pt --simplify --img-size 640 640 --dynamic-batch --grid

cd ..
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
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

Model   |BatchSize  |Precision |FPS       |MAP@0.5   |MAP@0.5:0.95 |
--------|-----------|----------|----------|----------|-------------|
yolov7  |    32     |   FP16   |341.681   |  0.695   |  0.509      |
yolov7  |    32     |   INT8   |669.783   |  0.685   |  0.473      |

## Reference

YOLOv7: <https://github.com/WongKinYiu/yolov7>
