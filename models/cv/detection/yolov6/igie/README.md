# YOLOv6

## Description

YOLOv6 integrates cutting-edge object detection advancements from industry and academia, incorporating recent innovations in network design, training strategies, testing techniques, quantization, and optimization methods. This culmination results in a suite of deployment-ready networks, accommodating varied use cases across different scales.  

## Setup

### Install
```
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install pycocotools
```

### Download

Pretrained model: <https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
# install yolov6
git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
pip3 install -r requirements.txt

# export onnx model
python3 deploy/ONNX/export_onnx.py --weights ../yolov6s.pt --img 640 --dynamic-batch --simplify

cd ..
```

## Inference
```bash
export DATASETS_DIR=/Path/to/coco/
```
### FP16

```bash
# Accuracy
bash scripts/infer_yolov6_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov6_fp16_performance.sh
```

## Results

Model    |BatchSize  |Precision |FPS       |MAP@0.5   |MAP@0.5:0.95 |
---------|-----------|----------|----------|----------|-------------|
yolov6   |    32     |   FP16   | 994.902  |  0.617   |   0.448     |

## Reference

YOLOv6: https://github.com/meituan/YOLOv6
