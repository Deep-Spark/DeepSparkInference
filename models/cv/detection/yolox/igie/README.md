# YOLOX

## Description

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities. 

## Setup

### Install
```
yum install mesa-libGL
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install pycocotools

#update gcc version
source /opt/rh/devtoolset-7/enable
```

### Download

Pretrained model: <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
# install yolox
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
python3 setup.py develop

# export onnx model
python3 tools/export_onnx.py -c ../yolox_m.pth -o 13 -n yolox-m --input input --output output --dynamic --output-name ../yolox.onnx

cd ..
```

## Inference
```bash
export DATASETS_DIR=/Path/to/coco/
```
### FP16

```bash
# Accuracy
bash scripts/infer_yolox_fp16_accuracy.sh
# Performance
bash scripts/infer_yolox_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_yolox_int8_accuracy.sh
# Performance
bash scripts/infer_yolox_int8_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS       |MAP@0.5   |
--------|-----------|----------|----------|----------|
yolox   |    32     |   FP16   |409.517   |  0.656   |
yolox   |    32     |   INT8   |844.991   |  0.637   |

## Reference

YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
