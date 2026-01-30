# Swin Transformer (ixRT)

## Model Description

Swin Transformer is a pioneering neural network architecture that introduces a novel approach to handling local and global information in computer vision tasks. Departing from traditional self-attention mechanisms, Swin Transformer adopts a hierarchical design, organizing its attention windows in a shifted manner. This innovation enables more efficient modeling of contextual information across different scales, enhancing the model's capability to capture intricate patterns.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/swin_s_model_sim.onnx>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
mkdir -p checkpoints
# download swin_s_model_sim.onnx into checkpoints
```

## Model Inference

```bash
export PROJ_DIR=./
export DATASETS_DIR=./imagenet-val/
export CHECKPOINTS_DIR=./checkpoints
export RUN_DIR=../../ixrt_common/
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
| :--------------: | :-------: | :-------: | :----:  | :------: | :------: |
| Swin Transformer |  32       | FP16      | 231.428 | 82.782   | 96.296   |
