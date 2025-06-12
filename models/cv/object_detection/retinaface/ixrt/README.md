# RetinaFace (IxRT)

## Model Description

RetinaFace is an efficient single-stage face detection model that employs a multi-task learning strategy to simultaneously predict facial locations, landmarks, and 3D facial shapes. It utilizes feature pyramids and context modules to extract multi-scale features and employs a self-supervised mesh decoder to enhance detection accuracy. RetinaFace demonstrates excellent performance on datasets like WIDER FACE, supports real-time processing, and its code and datasets are publicly available for researchers.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth>

Dataset: <http://shuoyang1213.me/WIDERFACE/> to download the validation dataset.

```bash
wget https://github.com/biubug6/Face-Detector-1MB-with-landmark/raw/master/weights/mobilenet0.25_Final.pth
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt

python3 setup.py build_ext --inplace
```

### Model Conversion

```bash
# export onnx model
python3 torch2onnx.py --model mobilenet0.25_Final.pth --onnx_model mnetv1_retinaface.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/widerface/
export GT_DIR=../igie/widerface_evaluate/ground_truth
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
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| RetinaFace | 32        | FP16      | 8536.367 | 80.84      | 69.34         | 37.31      |

## References

- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
