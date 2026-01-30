# UNet (ixRT)

## Model Description

UNet is a convolutional neural network architecture for image segmentation, featuring a symmetric encoder-decoder structure. The encoder gradually extracts features and reduces spatial dimensions, while the decoder restores resolution through upsampling. Key skip connections allow high-resolution features to be directly passed to the decoder, enhancing detail retention and segmentation accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/unet_export.onnx>

Dataset: <https://drive.grand-challenge.org/> to download the dataset.

### Install Dependencies

```bash
pip3 install onnxsim
pip3 install opencv-python==4.6.0.66
pip3 uninstall mmcv-full mmcv -y
pip3 install mmcv==1.5.3
pip3 install prettytable
pip3 install onnx
```

### Model Conversion

```bash
mkdir -p checkpoints
# deploy, will generate unet.onnx
python3 deploy.py  \
    --onnx_name checkpoints/unet_export.onnx \
    --save_dir checkpoints/ \
    --data_type float16 "$@"

mkdir -p datasets
# download DRIVE into datasets
```

## Model Inference

### FP16

```bash
# Accuracy
bash scripts/infer_unet_fp16_accuracy.sh
# Performance
bash scripts/infer_unet_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | aAcc |   mDice |  mAcc |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| UNet  | 1       | FP16      | 674.076 | 96.48 |  88.38 | 86.41 |

## References
