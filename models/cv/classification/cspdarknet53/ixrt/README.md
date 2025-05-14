# CSPDarkNet53 (IxRT)

## Model Description

CSPDarkNet53 is an enhanced convolutional neural network architecture that reduces redundant computations by integrating cross-stage partial network features and truncating gradient flow, thereby maintaining high accuracy while lowering computational costs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../ixrt_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
## cspdarknet50 is actually cspdarknet53
wget -O cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth

python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/cspnet/cspdarknet50_8xb32_in1k.py --weight cspdarknet53_3rdparty_8xb32_in1k_20220329-bd275287.pth --output cspdarknet53.onnx

# Use onnxsim optimize onnx model
mkdir -p checkpoints
onnxsim cspdarknet5.onnx checkpoints/cspdarknet53_sim.onnx

```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=/path/to/imagenet_val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
export CONFIG_DIR=../../ixrt_common/config/config/CSPDARKNET53_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_cspdarknet53_fp16_accuracy.sh 
# Performance
bash scripts/infer_cspdarknet53_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_cspdarknet53_int8_accuracy.sh 
# Performance
bash scripts/infer_cspdarknet53_int8_performance.sh
```

## Model Results

| Model        | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------------ | --------- | --------- | -------- | -------- | -------- |
| CSPDarkNet53 | 32        | FP16      | 3282.318 | 79.09    | 94.52    |
| CSPDarkNet53 | 32        | INT8      | 6335.86  | 75.49    | 92.66    |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
