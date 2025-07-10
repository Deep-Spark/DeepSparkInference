# FastReID (IGIE)

## Model Description

FastReID is a research platform that implements state-of-the-art re-identification algorithms.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth>

Dataset: <https://www.pkuml.org/resources/pku-vehicleid.html> to download the vehicleid dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# install fast-reid
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip3 install -r docs/requirements.txt

# export onnx model
python3 tools/deploy/onnx_export.py --config-file configs/VehicleID/bagtricks_R50-ibn.yml --name fast_reid --output ../ --opts MODEL.WEIGHTS ../vehicleid_bot_R50-ibn.pth
cd ..
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/VehicleID
```

### FP16

```bash
# Accuracy
bash scripts/infer_fastreid_fp16_accuracy.sh
# Performance
bash scripts/infer_fastreid_fp16_performance.sh
```

## Model Results

| Model    | BatchSize | Precision | FPS     | Rank-1(%) | Rank-5(%) | mAP   |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| FastReid | 32        | FP16      | 1850.78 | 88.39     | 98.45     | 92.79 |
