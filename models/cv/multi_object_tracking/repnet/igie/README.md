# RepNet-Vehicle-ReID (IGIE)

## Model Description

The paper "Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles" introduces a model named Deep Relative Distance Learning (DRDL), specifically designed for the problem of vehicle re-identification. DRDL employs a dual-branch deep convolutional network architecture, combined with a coupled clusters loss function and a mixed difference network structure, effectively mapping vehicle images into Euclidean space for similarity measurement.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/CaptainEven/RepNet-MDNet-VehicleReID>

Dataset: <https://www.pkuml.org/resources/pku-vehicleid.html> to download the VehicleID dataset.

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --weight epoch_14.pth --output repnet.onnx

# Use onnxsim optimize onnx model
onnxsim repnet.onnx repnet_opt.onnx
```

## Model Inference

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

## Model Results

| Model  | BatchSize | Precision | FPS      | Acc(%) |
|--------|-----------|-----------|----------|--------|
| RepNet | 32        | FP16      | 1373.579 | 99.88  |

## References

- [RepNet-MDNet-VehicleReID](https://github.com/CaptainEven/RepNet-MDNet-VehicleReID)
