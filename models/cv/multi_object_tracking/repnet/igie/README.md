# RepNet-Vehicle-ReID (IGIE)

## Model Description

The paper "Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles" introduces a model named Deep Relative Distance Learning (DRDL), specifically designed for the problem of vehicle re-identification. DRDL employs a dual-branch deep convolutional network architecture, combined with a coupled clusters loss function and a mixed difference network structure, effectively mapping vehicle images into Euclidean space for similarity measurement.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

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
| :----: | :----: | :----: | :----: | :----: |
| RepNet | 32        | FP16      | 1373.579 | 99.88  |

## References

- [RepNet-MDNet-VehicleReID](https://github.com/CaptainEven/RepNet-MDNet-VehicleReID)
