# FCOS (IxRT)

## Model Description

FCOS is an anchor-free model based on the Fully Convolutional Network (FCN) architecture for pixel-wise object detection. It implements a proposal-free solution and introduces the concept of centerness.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/1904.01355).

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth>

COCO2017: <https://cocodataset.org/>

- val2017: Path/To/val2017/*.jpg
- annotations: Path/To/annotations/instances_val2017.json

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

The inference of the FCOS model requires a dependency on a well-adapted mmcv-v1.7.0 library. Please inquire with the staff to obtain the relevant libraries.

You can follow the script [prepare_mmcv.sh](https://gitee.com/deep-spark/deepsparkhub/blob/master/toolbox/MMDetection/prepare_mmcv.sh) to build:

```bash
cd mmcv
sh build_mmcv.sh
sh install_mmcv.sh
```

### Model Conversion

MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project.It is utilized for model conversion. In MMDetection, Execute model conversion command, and the checkpoints folder needs to be created, (mkdir checkpoints) in project

```bash
mkdir -p checkpoints
git clone -b v2.25.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python3 tools/deployment/pytorch2onnx.py \
    /Path/to/fcos/ixrt/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py \
    checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth \
    --output-file /Path/To/ixrt/data/checkpoints/r50_fcos.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 800 800 \
    --show \
    --verify \
    --skip-postprocess \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1
```

If there are issues such as input parameter mismatch during model export, it may be due to ONNX version. To resolve this, please delete the last parameter (dynamic_slice) from the return value of the_slice_helper function in the /usr/local/lib/python3.10/site-packages/mmcv/onnx/onnx_utils/symbolic_helper.py file.

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/Path/to/coco/
export CHECKPOINTS_DIR=/Path/to/checkpoints
export RUN_DIR=./
```

### FP16

```bash
# Accuracy
bash scripts/infer_fcos_fp16_accuracy.sh
# Performance
bash scripts/infer_fcos_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS   | MAP@0.5 | MAP@0.5:0.95 |
| ----- | --------- | --------- | ----- | ------- | ------------ |
| FCOS  | 1         | FP16      | 51.62 | 0.546   | 0.360        |
