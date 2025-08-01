# ATSS (ixRT)

## Model Description

ATSS is an advanced adaptive training sample selection method that effectively enhances the performance of both anchor-based and anchor-free object detectors by dynamically choosing positive and negative samples based on the statistical characteristics of objects. The design of ATSS reduces reliance on hyperparameters, simplifies the sample selection process, and significantly improves detection accuracy without adding extra computational costs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth
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
python3 export.py --weight atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --cfg ../../ixrt_common/atss_r50_fpn_1x_coco.py --output checkpoints/atss.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/Path/to/coco/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common
```

### FP16

```bash
# Accuracy
bash scripts/infer_atss_fp16_accuracy.sh
# Performance
bash scripts/infer_atss_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ATSS  | 32        | FP16      | 133.573 | 0.541   | 0.367        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
