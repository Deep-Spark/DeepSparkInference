# YOLOv6

## Description

YOLOv6 integrates cutting-edge object detection advancements from industry and academia, incorporating recent innovations in network design, training strategies, testing techniques, quantization, and optimization methods. This culmination results in a suite of deployment-ready networks, accommodating varied use cases across different scales.  

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install pycocotools
pip3 install pycuda
```

### Download

Pretrained model: <https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
# get yolov6s.pt
wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
# set coco path
mkdir -p data/
ln -s /Path/to/coco/ data/coco
```

### Model Conversion

```bash
# install yolov6
git clone https://github.com/meituan/YOLOv6.git

pushd YOLOv6
pip3 install -r requirements.txt

# export onnx model
python3 deploy/ONNX/export_onnx.py --weights ../yolov6s.pt --img 640 --batch-size 32 --simplify
mv ../yolov6s.onnx ../data/

popd
```

## Inference

### FP16

```bash
# Accuracy
bash scripts/infer_yolov6s_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov6s_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov6s_int8_accuracy.sh
# Performance
bash scripts/infer_yolov6s_int8_performance.sh
```

## Results

| Model  | BatchSize | Precision | FPS      | MAP@0.5 |
| ------ | --------- | --------- | -------- | ------- |
| YOLOv6 | 32        | FP16      | 1107.511 | 0.355   |
| YOLOv6 | 32        | INT8      | 2080.475 | -       |

## Reference

YOLOv6: <https://github.com/meituan/YOLOv6>
