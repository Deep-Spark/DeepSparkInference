# CLIP (IxRT)

## Model Description

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/openai/clip-vit-base-patch32>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

```bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-base-patch32 clip-vit-base-patch32
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
Contact the Iluvatar administrator to get the missing packages:
- ixrt-1.0.0a0+corex.4.3.0.20250723-cp310-cp310-linux_x86_64.whl or later
```

### Model Conversion

```bash
mkdir -p checkpoints/clip
python3 export.py --output checkpoints/clip/clip.onnx
```

## Model Inference

```bash
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
export OPTIMIER_FILE=./iluvatar-corex-ixrt/tools/optimizer/optimizer.py
export DATASETS_DIR=/path/to/imagenet_val/
export PROJ_DIR=./
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common
export CONFIG_DIR=../../ixrt_common/config/CLIP_CONFIG
```

### FP16

```bash
# Accuracy
bash scripts/infer_clip_fp16_accuracy.sh
# Performance
bash scripts/infer_clip_fp16_performance.sh
```

## Model Results

| Model | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| CLIP  | 32        | FP16      | 350.94 | 59.68    | 86.14    |
