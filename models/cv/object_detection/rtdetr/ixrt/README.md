# RTDETR (ixRT)

## Model Description

RT-DETR is the first real-time end-to-end object detector. Specifically, we design an efficient hybrid encoder that effectively processes multi-scale features by decoupling intra-scale interaction and cross-scale fusion. Additionally, we propose an IoU-aware query selection mechanism to optimize the initialization of decoder queries. Moreover, RT-DETR supports flexible adjustment of inference speed by using a different number of decoder layers without requiring retraining, which greatly facilitates practical deployment of real-time object detectors.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only | 26.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/rtdetrv3_r18vd_6x_coco_image_sim.onnx>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.
- <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> to download the annotations dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip
unzip -q -d ./coco annotations_trainval2017.zip

coco
├── annotations
│   └── instances_train2017.json
│   └── instances_val2017.json
│   └── captions_train2017.json
│   └── captions_val2017.json
│   └── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
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
pip3 install tqdm
pip3 install onnx
pip3 install pycocotools
pip3 install opencv-python==4.6.0.66
```

### Model Conversion

```bash
mkdir -p checkpoints
# download onnx into checkpoints
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16
```bash
# Accuracy
bash scripts/infer_rtdetr_fp16_accuracy.sh
# Performance
bash scripts/infer_rtdetr_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision |  FPS  | IOU@0.5 | IOU@0.5:0.95 |
|:------:|:---------:|:---------:|:-----:|:-------:|:------------:|
| RT-DETR|     32    |   FP16    | 326.427 |  0.656  |     0.480    |
