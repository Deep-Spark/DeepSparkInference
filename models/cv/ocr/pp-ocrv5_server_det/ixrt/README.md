# PP-OCRv5_server_det_onnx (ixRT)

## Model Description

PP-OCRv5_server_det is one of the PP-OCRv5_det series, the latest generation of text detection models developed by the PaddleOCR team. Designed for high-performance applications, it supports the detection of text in diverse scenarios—including handwriting, vertical, rotated, and curved text—across multiple languages such as Simplified Chinese, Traditional Chinese, English, and Japanese. Key features include robust handling of complex layouts, varying text sizes, and challenging backgrounds, making it suitable for practical applications like document analysis, license plate recognition, and scene text detection.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| BI-V150 | 5.0.0 | 26.12 | release/26.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.12`

## Model Preparation

### Prepare Resources

Pretrained model: <https://modelscope.cn/models/PaddlePaddle/PP-OCRv5_server_det_onnx>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Conversion

```bash
mkdir checkpoints
python3 modify_batchsize.py --batch_size 1 --origin_model inference.onnx --output_model checkpoints/inference.onnx
```

## Model Inference

```bash
export CHECKPOINTS_DIR=./checkpoints
```

### FP16

```bash
# Performance
ixrtexec --onnx ${CHECKPOINTS_DIR}/inference.onnx --save_engine ${CHECKPOINTS_DIR}/inference.engine --precision fp16 --warmUp 1 --iterations 20
```

## Model Results

| Model                    | BatchSize | Precision | FPS    |
| ------------------------ | --------- | --------- | ------ |
| PP-OCRv5_server_det_onnx | 1         | FP16      | 40.954 |
