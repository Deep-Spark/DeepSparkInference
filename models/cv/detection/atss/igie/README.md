# ATSS

## Description

ATSS is an advanced adaptive training sample selection method that effectively enhances the performance of both anchor-based and anchor-free object detectors by dynamically choosing positive and negative samples based on the statistical characteristics of objects. The design of ATSS reduces reliance on hyperparameters, simplifies the sample selection process, and significantly improves detection accuracy without adding extra computational costs.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install onnx
pip3 install tqdm
pip3 install onnxsim
pip3 install mmdet==3.3.0
pip3 install mmdeploy==1.3.1
pip3 install mmengine==0.10.4
```

### Download
Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth> 

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.


### Model Conversion
```bash
# export onnx model
python3 export.py --weight atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --cfg atss_r50_fpn_1x_coco.py --output atss.onnx

# use onnxsim optimize onnx model
onnxsim atss.onnx atss_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_atss_fp16_accuracy.sh
# Performance
bash scripts/infer_atss_fp16_performance.sh
```

## Results

|   Model   | BatchSize | Input Shape | Precision |    FPS    | mAP@0.5(%) |
| :-------: | :-------: | :---------: | :-------: | :-------: | :--------: |
|   ATSS    |    32     |   800x800   |    FP16   |   81.671  |    0.541   |