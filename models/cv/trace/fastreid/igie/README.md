# FastReID

## Description

FastReID is a research platform that implements state-of-the-art re-identification algorithms.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth>

Dataset: <https://www.pkuml.org/resources/pku-vehicleid.html> to download the vehicleid dataset.

### Model Conversion

```bash
# install fast-reid
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip3 install -r requirements.txt

# export onnx model
python3 tools/deploy/onnx_export.py --config-file configs/VehicleID/bagtricks_R50-ibn.yml --name fast_reid --output ../ --opts MODEL.WEIGHTS ../vehicleid_bot_R50-ibn.pth
cd..
```

## Inference

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

## Results

Model    |BatchSize  |Precision |FPS       |Rank-1(%) |Rank-5(%) |mAP     |
---------|-----------|----------|----------|----------|----------|--------|
FastReid |    32     |   FP16   |  1850.78 |  88.39   |  98.45   | 92.79  |
