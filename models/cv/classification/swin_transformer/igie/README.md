# Swin Transformer (IGIE)

## Model Description

Swin Transformer is a pioneering neural network architecture that introduces a novel approach to handling local and global information in computer vision tasks. Departing from traditional self-attention mechanisms, Swin Transformer adopts a hierarchical design, organizing its attention windows in a shifted manner. This innovation enables more efficient modeling of contextual information across different scales, enhancing the model's capability to capture intricate patterns.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/docs/transformers/model_doc/swin>

```bash
git lfs install
git clone https://huggingface.co/microsoft/swin-tiny-patch4-window7-224 swin-tiny-patch4-window7-224
```

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --output swin_transformer.onnx

# Use onnxsim optimize onnx model
onnxsim swin_transformer.onnx swin_transformer_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_swin_transformer_fp16_accuracy.sh
# Performance
bash scripts/infer_swin_transformer_fp16_performance.sh
```

## Model Results

| Model            | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
|------------------|-----------|-----------|---------|----------|----------|
| Swin Transformer | 32        | FP16      | 1104.52 | 80.578   | 95.2     |
