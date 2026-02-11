# YOLOF (ixRT)

## Model Description

YOLOF is a lightweight object detection model that focuses on single-level feature maps for detection and enhances feature representation using dilated convolution modules. With a simple and efficient structure, it is well-suited for real-time object detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth>

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

- mmcv-*.whl

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir -p checkpoints/

# download the weight from the recommend link
wget https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth

# export onnx model
python3 export.py --weight yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth --cfg ../../ixrt_common/yolof_r50-c5_8xb8-1x_coco.py --output checkpoints/yolof.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=./coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolof_fp16_accuracy.sh
# Performance
bash scripts/infer_yolof_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----:| :-------: | :-------: | :----: | :-----: | :----------: |
| YOLOF | 32        | FP16      | 331.06 | 0.527   | 0.343        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)