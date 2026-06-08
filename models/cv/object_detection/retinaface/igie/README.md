# RetinaFace (IGIE)

## Model Description

RetinaFace is an efficient single-stage face detection model that employs a multi-task learning strategy to simultaneously predict facial locations, landmarks, and 3D facial shapes. It utilizes feature pyramids and context modules to extract multi-scale features and employs a self-supervised mesh decoder to enhance detection accuracy. RetinaFace demonstrates excellent performance on datasets like WIDER FACE, supports real-time processing, and its code and datasets are publicly available for researchers.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

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
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| RetinaFace | 32        | FP16      | 8304.626 | 80.13      | 68.52         | 36.59      |

## References

- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
