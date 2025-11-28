# YOLOv11 (IGIE)

## Model Description

YOLOv11 is the latest generation of the YOLO (You Only Look Once) series object detection model released by Ultralytics. Building upon the advancements of previous YOLO models, such as YOLOv5 and YOLOv8, YOLOv11 introduces comprehensive upgrades to further enhance performance, flexibility, and usability. It is a versatile deep learning model designed for multi-task applications, supporting object detection, instance segmentation, image classification, keypoint pose estimation, and rotated object detection.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>

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
pip3 install -r requirements.txt
```

## Model Conversion

```bash
python3 export.py --weight yolo11n.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov11_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov11_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_yolov11_int8_accuracy.sh
# Performance
bash scripts/infer_yolov11_int8_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------- | ------- | ------------ |
| YOLOv11 | 32        | FP16      | 1328.49 | 0.551   | 0.393        |
| YOLOv11 | 32        | INT8      | 1478.113 | 0.506   | 0.349        |

## References

YOLOv11: <https://docs.ultralytics.com/zh/models/yolo11/>
