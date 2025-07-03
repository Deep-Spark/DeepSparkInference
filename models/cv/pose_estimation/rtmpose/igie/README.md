# RTMPose (IGIE)

## Model Description

RTMPose, a state-of-the-art framework developed by Shanghai AI Laboratory, excels in real-time multi-person pose estimation by integrating an innovative model architecture with the efficiency of the MMPose foundation. The framework's architecture is meticulously designed to enhance performance and reduce latency, making it suitable for a variety of applications where real-time analysis is crucial.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth>

Dataset: <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.

```bash
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
# before install mmpose==1.3.1 need to install chchumpy==0.70 which is too older that is not compatible with newer Python versions or pip
# so need to downgrade pip to version 20.2.4
pip install pip==20.2.4
pip install mmpose==1.3.1
pip install --upgrade pip
pip install -r requirements.txt
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth --cfg rtmpose-m_8xb256-420e_coco-256x192.py --output rtmpose.onnx

# use onnxsim optimize onnx model
onnxsim rtmpose.onnx rtmpose_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_rtmpose_fp16_accuracy.sh
# Performance
bash scripts/infer_rtmpose_fp16_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| RTMPose | 32        | FP16      | 2313.33 | 0.936   | 0.773        |

## References

- [mmpose](https://github.com/open-mmlab/mmpose.git)
