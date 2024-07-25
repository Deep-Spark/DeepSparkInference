# FSAF

## Description

The FSAF (Feature Selective Anchor-Free) module is an innovative component for single-shot object detection that enhances performance through online feature selection and anchor-free branches. The FSAF module dynamically selects the most suitable feature level for each object instance, rather than relying on traditional anchor-based heuristic methods. This improvement significantly boosts the accuracy of object detection, especially for small targets and in complex scenes. Moreover, compared to existing anchor-based detectors, the FSAF module maintains high efficiency while adding negligible additional inference overhead.

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

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight fsaf_r50_fpn_1x_coco-94ccc51f.pth --cfg fsaf_r50_fpn_1x_coco.py --output fsaf.onnx

# use onnxsim optimize onnx model
onnxsim fsaf.onnx fsaf_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_fsaf_fp16_accuracy.sh
# Performance
bash scripts/infer_fsaf_fp16_performance.sh
```

## Results

| Model | BatchSize | Input Shape | Precision |   FPS   | mAP@0.5(%) |
| :---: | :-------: | :---------: | :-------: | :-----: | :--------: |
| FSAF  |    32     |   800x800   |   FP16    | 178.748 |   0.530    |
