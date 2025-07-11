# CLIP (IGIE)

## Model Description

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

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
```

### Model Conversion

```bash
python3 export.py --output clip.onnx

# Use onnxsim optimize onnx model
onnxsim clip.onnx clip_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
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
| CLIP  | 32        | FP16      | 496.91 | 59.68    | 86.16    |
