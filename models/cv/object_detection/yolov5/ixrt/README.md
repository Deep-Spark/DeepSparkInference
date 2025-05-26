# YOLOv5-m (IxRT)

## Model Description

The YOLOv5 architecture is designed for efficient and accurate object detection tasks in real-time scenarios. It employs a single convolutional neural network to simultaneously predict bounding boxes and class probabilities for multiple objects within an image. The YOLOV5m is a medium-sized model.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

- val2017: Path/To/val2017/*.jpg
- annotations: Path/To/annotations/instances_val2017.json

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

mkdir checkpoints
git clone https://github.com/ultralytics/yolov5
# 切换到需要的版本分支
git checkout v6.1

# 有一些环境需要安装
wget https://ultralytics.com/assets/Arial.ttf
cp Arial.ttf  /root/.config/Ultralytics/Arial.ttf

# 转换为onnx (具体实现可以参考 export.py 中的 export_onnx 函数)
python3 export.py --weights yolov5m.pt --include onnx --opset 11 --batch-size 32
mv yolov5m.onnx /Path/to/checkpoints
```

## Model Inference

```bash
export PROJ_DIR=/Path/to/yolov5/ixrt
export DATASETS_DIR=/Path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=/Path/to/yolov5/ixrt
export CONFIG_DIR=config/YOLOV5_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov5_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov5_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov5_int8_accuracy.sh
# Performance
bash scripts/infer_yolov5_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv5 | 32        | FP16      | 680.93  | 0.637   | 0.447        |
| YOLOv5 | 32        | INT8      | 1328.50 | 0.627   | 0.425        |
