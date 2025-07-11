# Conformer Base (IGIE)

## Model Description

Conformer is a novel network architecture that addresses the limitations of conventional Convolutional Neural Networks (CNNs) and visual transformers.  Rooted in the Feature Coupling Unit (FCU), Conformer efficiently fuses local features and global representations at different resolutions through interactive processes. Its concurrent architecture ensures the maximal retention of both local and global features. 

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://drive.google.com/file/d/1oeQ9LSOGKEUaYGu7WTlUGl3KDsQIi0MA/view?usp=sharing>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight conformer_base_patch16.pth --output conformer_base.onnx

# Use onnxsim optimize onnx model
onnxsim conformer_base.onnx conformer_base_opt.onnx

```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_conformer_base_fp16_accuracy.sh
# Performance
bash scripts/infer_conformer_base_fp16_performance.sh
```

## Model Results

| Model          | BatchSize | Precision | FPS    | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Conformer Base | 32        | FP16      | 428.73 | 83.83    | 96.59    |

## References

- [Conformer](https://github.com/pengzhiliang/Conformer)
