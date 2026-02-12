# YOLOv5s (ixRT)

## Model Description

The YOLOv5 architecture is designed for efficient and accurate object detection tasks in real-time scenarios. It employs a single convolutional neural network to simultaneously predict bounding boxes and class probabilities for multiple objects within an image. The YOLOV5s is a tiny model.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
├── images
│   ├── train2017
│   └── val2017
├── labels
│   ├── train2017
│   └── val2017
├── LICENSE
├── README.txt
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
git clone -b v6.1 --depth 1 https://github.com/ultralytics/yolov5

# 有一些环境需要安装
wget https://ultralytics.com/assets/Arial.ttf
cp Arial.ttf  /root/.config/Ultralytics/Arial.ttf

# 转换为onnx (具体实现可以参考 export.py 中的 export_onnx 函数)
pushd ./yolov5
# set weights_only=False to be comaptible with pytorch 2.7 
sed -i '96 s/map_location)/map_location, weights_only=False)/' ./models/experimental.py
# download the weight from the recommend link
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python3 export.py --weights yolov5s.pt --include onnx --opset 11 --batch-size 32
mv yolov5s.onnx ../checkpoints
popd
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/Path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/images/val2017
export RUN_DIR=../../ixrt_common
export CONFIG_DIR=../../ixrt_common/config/YOLOV5S_CONFIG
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
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv5s | 32        | FP16      | 1112.66 | 0.565   | 0.370        |
| YOLOv5s | 32        | INT8      | 2440.54 | 0.557   | 0.351        |
