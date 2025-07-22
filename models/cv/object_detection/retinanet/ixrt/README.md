# RetinaNet (IxRT)

## Model Description

RetinaNet, an innovative object detector, challenges the conventional trade-off between speed and accuracy in the realm of computer vision. Traditionally, two-stage detectors, exemplified by R-CNN, achieve high accuracy by applying a classifier to a limited set of candidate object locations. In contrast, one-stage detectors, like RetinaNet, operate over a dense sampling of possible object locations, aiming for simplicity and speed.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
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
python3 export.py --weight retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --cfg ../../ixrt_common/retinanet_r50_fpn_1x_coco.py --output checkpoints/retinanet.onnx
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
bash scripts/infer_retinanet_fp16_accuracy.sh
# Performance
bash scripts/infer_retinanet_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| PAA  | 32        | FP16      | 162.171 | 0.515   | 0.335       |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
