# YOLOv4 (ixRT)

## Model Description

YOLOv4 employs a two-step process, involving regression for bounding box positioning and classification for object categorization. it amalgamates past YOLO family research contributions with novel features like WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, DropBlock regularization, and CIoU loss.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained cfg: <https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg>
Pretrained model: <https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights>

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

Contact the Iluvatar administrator to get the missing packages:
- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

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
# clone yolov4
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git yolov4

# download weight
mkdir checkpoints
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P checkpoints

# export onnx model
python3 export.py --cfg yolov4/cfg/yolov4.cfg --weight yolov4.weights --output yolov4.onnx
mv yolov4.onnx checkpoints/yolov4.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=./coco/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=./coco/annotations/instances_val2017.json
export EVAL_DIR=./coco/images/val2017
export RUN_DIR=./
export CONFIG_DIR=config/YOLOV4_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov4_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov4_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov4_int8_accuracy.sh
# Performance
bash scripts/infer_yolov4_int8_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS    | MAP@0.5 |
| ------ | --------- | --------- | ------ | ------- |
| YOLOv4 | 32        | FP16      | 303.27 | 0.730   |
| YOLOv4 | 32        | INT8      | 682.14 | 0.608   |

## References

- [darknet](https://github.com/AlexeyAB/darknet)
- [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
