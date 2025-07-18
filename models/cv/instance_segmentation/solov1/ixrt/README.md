# SOLOv1 (IxRT)

## Model Description

SOLO (Segmenting Objects by Locations) is a new instance segmentation method that differs from traditional approaches by introducing the concept of “instance categories”. Based on the location and size of each instance, SOLO assigns each pixel to a corresponding instance category. This method transforms the instance segmentation problem into a single-shot classification task, simplifying the overall process.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_3x_coco/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

The inference of the Solov1 model requires a dependency on a well-adapted mmcv-v1.7.0 library. Please contact the Iluvatar administrator to get the missing packages:
- mmcv_full-1.7.0+corex.20250108131027-cp310-cp310-linux_x86_64.whl

or You can follow the script [prepare_mmcv.sh](https://gitee.com/deep-spark/deepsparkhub/blob/master/toolbox/MMDetection/prepare_mmcv.sh) to build:

```bash
cd mmcv
sh build_mmcv.sh
sh install_mmcv.sh
```

### Model Conversion

```bash
mkdir checkpoints
python3 solo_torch2onnx.py --cfg /path/to/solo/solo_r50_fpn_3x_coco.py --checkpoint /path/to/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth --batch_size 1
mv r50_solo_bs1_800x800.onnx /Path/to/checkpoints/r50_solo_bs1_800x800.onnx
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/coco2017/
export CHECKPOINTS_DIR=./checkpoints
export COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
export EVAL_DIR=${DATASETS_DIR}/val2017
export RUN_DIR=./
```

### FP16

```bash
# Accuracy
bash scripts/infer_solov1_fp16_accuracy.sh
# Performance
bash scripts/infer_solov1_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS   | MAP@0.5 | MAP@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| SOLOv1 | 1         | FP16      | 24.67 | 0.541   | 0.338        |
