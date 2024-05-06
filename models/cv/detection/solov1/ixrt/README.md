# Solov1

## Description
SOLO (Segmenting Objects by Locations) is a new instance segmentation method that differs from traditional approaches by introducing the concept of “instance categories”. Based on the location and size of each instance, SOLO assigns each pixel to a corresponding instance category. This method transforms the instance segmentation problem into a single-shot classification task, simplifying the overall process.

## Setup

### Install
```bash
yum install mesa-libGL

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install mmdet==2.28.2
pip3 install addict
pip3 install yapf
```

### Dependency
The inference of the Solov1 model requires a dependency on a well-adapted mmcv-v1.7.0 library. Please inquire with the staff to obtain the relevant libraries.
```bash
cd mmcv
sh build_mmcv.sh
sh install_mmcv.sh
```

### Download
Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_3x_coco/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

### Model Conversion
```bash
mkdir checkpoints
python3 solo_torch2onnx.py --cfg /path/to/solo/solo_r50_fpn_3x_coco.py --checkpoint /path/to/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth --batch_size 1
mv r50_solo_bs1_800x800.onnx /Path/to/checkpoints/r50_solo_bs1_800x800.onnx
```

## Inference
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

## Results 

Model   |BatchSize  |Precision |FPS       |MAP@0.5   |MAP@0.5:0.95
--------|-----------|----------|----------|----------|------------
Solov1  |    1      |   FP16   | 24.67    |  0.541   | 0.338