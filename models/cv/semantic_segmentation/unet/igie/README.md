# UNet (IGIE)

## Model Description

UNet is a convolutional neural network architecture for image segmentation, featuring a symmetric encoder-decoder structure. The encoder gradually extracts features and reduces spatial dimensions, while the decoder restores resolution through upsampling. Key skip connections allow high-resolution features to be directly passed to the decoder, enhancing detail retention and segmentation accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.06 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth>

Dataset: <https://www.cityscapes-dataset.com> to download the dataset.

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
pip3 install -r requirements.txt
pip3 install mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
```

### Model Conversion

```bash
# export onnx model

python3 export.py --weight fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth --cfg fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py --output unet.onnx

# use onnxsim optimize onnx model
onnxsim unet.onnx unet_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/cityscapes/
```

### FP16

```bash
# Accuracy
bash scripts/infer_unet_fp16_accuracy.sh
# Performance
bash scripts/infer_unet_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    |  mIoU   |
| :----: | :----: | :----: | :----: | :----: |
| UNet  | 16        | FP16      | 66.265 |  69.48  |

## References

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
