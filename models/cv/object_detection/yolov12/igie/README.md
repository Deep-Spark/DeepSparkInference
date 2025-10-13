# YOLOv12 (IGIE)

## Model Description

YOLOv12 achieves high precision and efficient real-time object detection by integrating attention mechanisms and innovative architectural design. YOLOv12-N is the lightweight version of this series, optimized for resource-constrained environments, maintaining the core advantages of YOLOv12 while offering fast inference and excellent detection accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt>

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
git clone --depth 1 https://github.com/sunsmarterjie/yolov12.git
cd yolov12
pip3 install -e .
cd ..

python3 export.py --weight yolov12n.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov12_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov12_fp16_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------- | ------- | ------------ |
| YOLOv12 | 32        | FP16      | 666.641 | 0.559   | 0.403        |

## References

- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
