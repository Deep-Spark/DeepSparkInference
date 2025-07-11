# YOLOv7 (IxRT)

## Model Description

YOLOv7 is an object detection model based on the YOLO (You Only Look Once) series. It is an improved version of YOLOv5 developed by the Ultralytics team. YOLOv7 aims to enhance the performance and efficiency of object detection through a series of improvements including network architecture, training strategies, and data augmentation techniques, in order to achieve more accurate and faster object detection.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt>

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

pip3 install -r ../../ixrt_common/requirements.txt
```

### Model Conversion

```bash

git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python3 export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 --batch-size 16
mkdir /Your_Projects/To/checkpoints
mv yolov7.onnx /Path/to/checkpoints/yolov7m.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/Path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/images/val2017
export RUN_DIR=../../ixrt_common
export CONFIG_DIR=../../ixrt_common/config/YOLOV7_CONFIG
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

## Model Results

| Model  | BatchSize | Precision | FPS    | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOv7 | 32        | FP16      | 375.41 | 0.693   | 0.506        |
| YOLOv7 | 32        | INT8      | 816.71 | 0.688   | 0.471        |

## References

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
