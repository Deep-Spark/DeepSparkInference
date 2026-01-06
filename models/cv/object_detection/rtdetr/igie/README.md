# RTDetr (IGIE)

## Model Description

Real-Time Detection Transformer (RT-DETR), developed by Baidu, is a cutting-edge end-to-end object detector that provides real-time performance while maintaining high accuracy. It is based on the idea of DETR (the NMS-free framework), meanwhile introducing conv-based backbone and an efficient hybrid encoder to gain real-time speed. 

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only | 26.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/rtdetrv3_r18vd_6x_coco_image.onnx>

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
pip3 install -r requirements.txt
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

### INT8

```bash
# Accuracy
bash scripts/infer_rtdetr_int8_accuracy.sh
# Performance
bash scripts/infer_rtdetr_int8_performance.sh
```

## Model Results

|  Model  | BatchSize | Precision | FPS     | MAP@0.5 |
| :-----: | :----: | :----: | :----: | :----: |
| RTDetr | 32        | FP16      | 335.8 | 0.654   |
| RTDetr | 32        | INT8      | 298.2 | 0.651   |

