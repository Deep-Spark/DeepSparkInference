# GoogleNet

## Description

Introduced in 2014, GoogleNet revolutionized image classification models by introducing the concept of inception modules. These modules utilize parallel convolutional filters of different sizes, allowing the network to capture features at various scales efficiently. With its emphasis on computational efficiency and the reduction of parameters, GoogleNet achieved competitive accuracy while maintaining a relatively low computational cost. 

## Setup

### Install
```
pip3 install onnx
pip3 install tqdm
```

### Download

Pretrained model: <https://download.pytorch.org/models/googlenet-1378be20.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion
```bash
python3 export.py --weight googlenet-1378be20.pth --output googlenet.onnx
```

## Inference
```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```
### FP16

```bash
# Accuracy
bash scripts/infer_googlenet_fp16_accuracy.sh
# Performance
bash scripts/infer_googlenet_fp16_performance.sh
```

### INT8
```bash
# Accuracy
bash scripts/infer_googlenet_int8_accuracy.sh
# Performance
bash scripts/infer_googlenet_int8_performance.sh
```

## Results

Model     |BatchSize  |Precision |FPS       |Top-1(%) |Top-5(%)
----------|-----------|----------|----------|---------|--------
GoogleNet |    32     |   FP16   | 6564.20  |  62.44  | 84.31
GoogleNet |    32     |   INT8   | 7910.65  |  61.06  | 83.26
