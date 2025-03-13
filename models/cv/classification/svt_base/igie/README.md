# SVT Base (IGIE)

## Model Description

SVT Base is a mid-sized variant of the Sparse Vision Transformer (SVT) series, designed to combine the expressive power of Vision Transformers (ViTs) with the efficiency of sparse attention mechanisms. By employing sparse attention and multi-stage feature extraction, SVT-Base reduces computational complexity while retaining global modeling capabilities.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 export.py --cfg mmpretrain/configs/twins/twins-svt-base_8xb128_in1k.py --weight twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth --output svt_base.onnx

# Use onnxsim optimize onnx model
onnxsim svt_base.onnx svt_base_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_svt_base_fp16_accuracy.sh
# Performance
bash scripts/infer_svt_base_fp16_performance.sh
```

## Model Results

| Model     | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------| --------- | --------- | -------- | -------- | -------- |
| SVT Base  | 32        | FP16      | 673.165  | 82.865   | 96.213   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
