# RetinaFace (IGIE)

## Model Description

RetinaFace is an efficient single-stage face detection model that employs a multi-task learning strategy to simultaneously predict facial locations, landmarks, and 3D facial shapes. It utilizes feature pyramids and context modules to extract multi-scale features and employs a self-supervised mesh decoder to enhance detection accuracy. RetinaFace demonstrates excellent performance on datasets like WIDER FACE, supports real-time processing, and its code and datasets are publicly available for researchers.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth>

Dataset: <http://shuoyang1213.me/WIDERFACE/> to download the validation dataset.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

```bash
wget https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight mobilenet0.25_Final.pth --output retinaface.onnx

# use onnxsim optimize onnx model
onnxsim retinaface.onnx retinaface_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/widerface/
```

### FP16

```bash
# Accuracy
bash scripts/infer_retinaface_fp16_accuracy.sh
# Performance
bash scripts/infer_retinaface_fp16_performance.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS      | Easy AP(%) | Medium AP (%) | Hard AP(%) |
|------------|-----------|-----------|----------|------------|---------------|------------|
| RetinaFace | 32        | FP16      | 8304.626 | 80.13      | 68.52         | 36.59      |

## References

- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
