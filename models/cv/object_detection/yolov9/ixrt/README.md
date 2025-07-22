# YOLOv9 (IxRT)

## Model Description

YOLOv9 represents a major leap in real-time object detection by introducing innovations like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN), significantly improving efficiency, accuracy, and adaptability. Developed by an open-source team and building on the YOLOv5 codebase, it sets new benchmarks on the MS COCO dataset. YOLOv9's architecture effectively addresses information loss in deep neural networks, enhancing learning capacity and ensuring higher detection accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt>

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

## Model Conversion

```bash
mkdir -p checkpoints/
mv yolov9s.pt yolov9.pt
python3 export.py --weight yolov9.pt --batch 32
onnxsim yolov9.onnx ./checkpoints/yolov9.onnx
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
bash scripts/infer_yolov9_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov9_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| ------ | --------- | --------- | ------ | ------- | ------------ |
| YOLOv9 | 32        | FP16      | 814.42 | 0.625   | 0.464        |

## References

- [YOLOv9](https://docs.ultralytics.com/models/yolov9)
