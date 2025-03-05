# YOLOv5s (IxRT)

## Model Description

The YOLOv5 architecture is designed for efficient and accurate object detection tasks in real-time scenarios. It employs a single convolutional neural network to simultaneously predict bounding boxes and class probabilities for multiple objects within an image. The YOLOV5s is a tiny model.

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir -p checkpoints
git clone https://github.com/ultralytics/yolov5
# 切换到需要的版本分支
pushd yolov5/
git checkout v6.1

# 有一些环境需要安装
wget https://ultralytics.com/assets/Arial.ttf
mkdir -p /root/.config/Ultralytics
cp Arial.ttf  /root/.config/Ultralytics/Arial.ttf

# 转换为onnx (具体实现可以参考 export.py 中的 export_onnx 函数)
python3 export.py --weights yolov5s.pt --include onnx --opset 11 --batch-size 32
mv yolov5s.onnx ../checkpoints
popd
```

## Model Inference

```bash
export PROJ_DIR=/Path/to/yolov5s/ixrt
export DATASETS_DIR=/Path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=${PROJ_DIR}/
export CONFIG_DIR=config/YOLOV5S_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov5s_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov5s_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov5s_int8_accuracy.sh
# Performance
bash scripts/infer_yolov5s_int8_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS     | MAP@0.5 | MAP@0.5:0.95 |
|---------|-----------|-----------|---------|---------|--------------|
| YOLOv5s | 32        | FP16      | 1112.66 | 0.565   | 0.370        |
| YOLOv5s | 32        | INT8      | 2440.54 | 0.557   | 0.351        |
