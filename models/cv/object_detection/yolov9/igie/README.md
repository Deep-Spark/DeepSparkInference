# YOLOv9 (IGIE)

## Model Description

YOLOv9 represents a major leap in real-time object detection by introducing innovations like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN), significantly improving efficiency, accuracy, and adaptability. Developed by an open-source team and building on the YOLOv5 codebase, it sets new benchmarks on the MS COCO dataset. YOLOv9's architecture effectively addresses information loss in deep neural networks, enhancing learning capacity and ensuring higher detection accuracy.

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt>

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Conversion

```bash
python3 export.py --weight yolov9s.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
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
