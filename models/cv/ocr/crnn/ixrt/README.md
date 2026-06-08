# CRNN (ixRT)

## Model Description

The CRNN (Convolutional Recurrent Neural Network) is a deep learning model specifically designed for sequence recognition tasks, and it is widely used in **optical character recognition **(OCR)—for example, in end-to-end recognition of text in natural scenes, handwriting, or variable-length text.

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

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/crnn.onnx>

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/crnn_sim_end.onnx>

### Install Dependencies

```bash
pip3 install tqdm onnxsim scikit-learn
```

## Model Conversion

```bash
mkdir -p checkpoints
# download crnn.onnx and crnn_sim_end.onnx into checkpoints
```

## Model Inference

### FP16

```bash
# Accuracy
bash scripts/infer_crnn_fp16_accuracy.sh
# Performance
bash scripts/infer_crnn_fp16_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS    | cos sim  | max diff |
| ------- | --------- | --------- | ------ | ------ |------ |
| CRNN    | 1         | FP16      | 21.248 | 0.999 |  0.038   |

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/algorithm/text_recognition/algorithm_rec_crnn.md)
