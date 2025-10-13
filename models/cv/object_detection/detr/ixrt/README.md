# DETR (ixRT)

## Model Description

DETR (DEtection TRansformer) is a novel approach that views object detection as a direct set prediction problem. This method streamlines the detection process, eliminating the need for many hand-designed components like non-maximum suppression procedures or anchor generation, which are typically used to explicitly encode prior knowledge about the task.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth>

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
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

```bash
mkdir checkpoints
python3 export_model.py --torch_file /path/to/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth --onnx_file checkpoints/detr_res50.onnx --bsz 1
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=./
export CONFIG_DIR=config/DETR_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_detr_fp16_accuracy.sh
# Performance
bash scripts/infer_detr_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS   | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| DETR  | 1         | FP16      | 65.84 | 0.370   | 0.198        |
