# RepNet-Vehicle-ReID

## Description

The paper "Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles" introduces a model named Deep Relative Distance Learning (DRDL), specifically designed for the problem of vehicle re-identification. DRDL employs a dual-branch deep convolutional network architecture, combined with a coupled clusters loss function and a mixed difference network structure, effectively mapping vehicle images into Euclidean space for similarity measurement.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
pip3 install onnxsim
```

### Download

Pretrained model: <https://github.com/CaptainEven/RepNet-MDNet-VehicleReID>

Dataset: <https://www.pkuml.org/resources/pku-vehicleid.html> to download the VehicleID dataset.

### Model Conversion

```bash
python3 export.py --weight epoch_14.pth --output repnet.onnx

# Use onnxsim optimize onnx model
onnxsim repnet.onnx repnet_opt.onnx
```

## Inference

```bash
export DATASETS_DIR=/Path/to/VehicleID/
```

### FP16

```bash
# Accuracy
bash scripts/infer_repnet_fp16_accuracy.sh
# Performance
bash scripts/infer_repnet_fp16_performance.sh
```

## Results

Model   |BatchSize  |Precision |FPS       |Acc(%)    |
--------|-----------|----------|----------|----------|
RepNet  |    32     |   FP16   |1373.579  |  99.88   |

## Reference

RepNet-MDNet-VehicleReID: <https://github.com/CaptainEven/RepNet-MDNet-VehicleReID>
