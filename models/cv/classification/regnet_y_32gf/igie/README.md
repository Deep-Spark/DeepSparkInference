# RegNet_y_32gf (IGIE)

## Model Description

RegNet_y_32gf is a variant from Facebook AI's RegNet series, belonging to the RegNet_y sub-series. Compared to the RegNet_x series, the RegNet_y series introduces SE (Squeeze-and-Excitation) modules into the network architecture, further enhancing feature representation capabilities.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name regnet_y_32gf --weight regnet_y_32gf-4dee3f7a.pth --output regnet_y_32gf.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_regnet_y_32gf_fp16_accuracy.sh
# Performance
bash scripts/infer_regnet_y_32gf_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :------------: | :-------: | :-------: | :-----: | :------: | :------: |
| RegNet_y_32gf  | 32        | FP16      | 413.726 | 80.832   | 95.321   |
