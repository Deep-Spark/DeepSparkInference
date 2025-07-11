# DeepSort (IGIE)

## Model Description

DeepSort integrates deep neural networks with traditional tracking methods to achieve robust and accurate tracking of objects in video streams. The algorithm leverages a combination of a deep appearance feature extractor and the Hungarian algorithm for data association.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model(ckpt.t7): <https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6>

Dataset: <https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html> to download the market1501 dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight ckpt.t7 --output deepsort.onnx

# Use onnxsim optimize onnx model
onnxsim deepsort.onnx deepsort_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/market1501/
```

### FP16

```bash
# Accuracy
bash scripts/infer_deepsort_fp16_accuracy.sh
# Performance
bash scripts/infer_deepsort_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_deepsort_int8_accuracy.sh
# Performance
bash scripts/infer_deepsort_int8_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS      | Acc(%) |
| :----: | :----: | :----: | :----: | :----: |
| DeepSort | 32        | FP16      | 17164.67 | 99.32  |
| DeepSort | 32        | INT8      | 20399.12 | 99.29  |
