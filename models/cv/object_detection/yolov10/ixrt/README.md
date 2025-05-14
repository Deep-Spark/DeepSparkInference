# YOLOv10 (IXRT)

## Model Description

YOLOv10, built on the Ultralytics Python package by researchers at Tsinghua University, introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt>

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

## Model Conversion

```bash
mkdir checkpoints
mv yolov10s.pt yolov10.pt
python3 export.py --weight yolov10.pt --batch 32
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov10_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov10_fp16_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------ | ------- | ------------ |
| YOLOv10 | 32        | FP16      | 810.97 | 0.629   | 0.461        |

## References

- [YOLOv10](https://docs.ultralytics.com/models/yolov10)
