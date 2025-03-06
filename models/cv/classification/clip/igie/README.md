# CLIP (IGIE)

## Model Description

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/docs/transformers/model_doc/clip>

```bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-base-patch32 clip-vit-base-patch32
```

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

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
|-------|-----------|-----------|--------|----------|----------|
| CLIP  | 32        | FP16      | 496.91 | 59.68    | 86.16    |
