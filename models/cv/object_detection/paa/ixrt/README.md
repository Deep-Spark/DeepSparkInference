# PAA (ixRT)

## Model Description

PAA (Probabilistic Anchor Assignment) is an algorithm for object detection that adaptively assigns positive and negative anchor samples using a probabilistic model. It employs a Gaussian mixture model to dynamically select positive and negative samples based on score distribution, avoiding the misassignment issues of traditional IoU threshold-based methods. PAA enhances detection accuracy, particularly in complex scenarios, and is compatible with existing detection frameworks.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 | release/26.09 |
| MR-V100 | 4.4.0 | 26.06 | release/26.06 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.09`

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth>

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

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth
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
# export onnx model
python3 export.py --weight paa_r50_fpn_1x_coco_20200821-936edec3.pth --cfg ../../ixrt_common/paa_r50_fpn_1x_coco.py --output checkpoints/paa.onnx
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
bash scripts/infer_paa_fp16_accuracy.sh
# Performance
bash scripts/infer_paa_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| PAA  | 32        | FP16      | 133.597 | 0.555   | 0.381        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
