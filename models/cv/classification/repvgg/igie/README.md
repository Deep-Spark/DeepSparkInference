# RepVGG (IGIE)

## Model Description

RepVGG is an innovative convolutional neural network architecture that combines the simplicity of VGG-style inference with a multi-branch topology during training. Through structural re-parameterization, RepVGG achieves high accuracy while significantly improving computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r ../../igie_common/requirements.txt
pip3 install http://files.deepspark.org.cn:880/deepspark/add-ons/mmcv_full-1.7.0+corex.20250108131027-cp310-cp310-linux_x86_64.whl
pip3 install mmcls==0.24.0 mmengine
# set weights_only=False to be comaptible with pytorch 2.7 
sed -i '334 s/map_location)/map_location, weights_only=False)/' /usr/local/lib/python3.10/site-packages/mmcv/runner/checkpoint.py
```

### Model Conversion

```bash
# git clone mmpretrain
git clone -b v0.24.0 https://github.com/open-mmlab/mmpretrain.git

# export onnx model
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py --weight repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth --output repvgg.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_repvgg_fp16_accuracy.sh
# Performance
bash scripts/infer_repvgg_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ------ | --------- | --------- | -------- | -------- | -------- |
| RepVGG | 32        | FP16      | 7423.035 | 72.345   | 90.543   |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
