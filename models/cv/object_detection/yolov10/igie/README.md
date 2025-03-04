# YOLOv10

## Model Description

YOLOv10, built on the Ultralytics Python package by researchers at Tsinghua University, introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Prepare Resources

Pretrained model: <https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt>

## Model Conversion

```bash
git clone --depth 1 https://github.com/THU-MIG/yolov10.git
cd yolov10
pip3 install -e . --no-deps
cd ..

python3 export.py --weight yolov10s.pt --batch 32

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
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

YOLOv10: <https://docs.ultralytics.com/models/yolov10/>
