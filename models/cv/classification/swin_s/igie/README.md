# Swin-S (IGIE)

## Model Description

Swin-S is a hierarchical vision Transformer that computes self-attention within local windows and uses shifted window partitioning to enable cross-window connections. It offers stronger representation capacity than Swin-T and is widely used as a backbone for classification, detection, and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/swin_s-5e29d889.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
```

### Model Conversion

```bash
python3 ../../igie_common/export.py --model-name swin_s --weight swin_s-5e29d889.pth --output swin_s.onnx
```

## Model Inference

```bash
export PATH="/opt/sw_home/local/corex/bin:${PATH}"
export CUDA_PATH=/opt/sw_home/local/corex
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/opt/sw_home/local/corex/lib64:/opt/sw_home/local/lib64:${LD_LIBRARY_PATH:-}"
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_swin_s_fp16_accuracy.sh
# Performance
bash scripts/infer_swin_s_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Swin-S | 32        | FP16      | 715.51 | 82.752   | 96.259   |
