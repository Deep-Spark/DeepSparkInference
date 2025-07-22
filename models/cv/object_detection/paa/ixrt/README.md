# PAA (IxRT)

## Model Description

PAA (Probabilistic Anchor Assignment) is an algorithm for object detection that adaptively assigns positive and negative anchor samples using a probabilistic model. It employs a Gaussian mixture model to dynamically select positive and negative samples based on score distribution, avoiding the misassignment issues of traditional IoU threshold-based methods. PAA enhances detection accuracy, particularly in complex scenarios, and is compatible with existing detection frameworks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

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
