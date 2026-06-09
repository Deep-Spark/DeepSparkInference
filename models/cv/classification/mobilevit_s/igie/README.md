# Mobilevit_s (IGIE)

## Model Description

The MobileViT-S model is a light-weight, general-purpose vision transformer designed specifically for mobile devices. It introduces a novel perspective by treating Transformers as convolutions, effectively combining the local processing strengths of CNNs with the global representation capabilities of Transformers.

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

Pretrained model: <https://huggingface.co/timm/mobilevit_s.cvnets_in1k>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Install Dependencies

```bash
pip3 install -r ../../igie_common/requirements.txt
pip3 install timm
```

### Model Conversion

```bash
# downloand mobilevit_s.cvnets_in1k from huggingface into ./mobilevit_s.cvnets_in1k
# export onnxmodel from timm
python3 export.py --model-name mobilevit_s.cvnets_in1k --output mobilevit_s.onnx

# use onnxsim optimize onnx model
onnxsim mobilevit_s.onnx mobilevit_s_opt.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
export RUN_DIR=../../igie_common/
```

### FP16

```bash
# Accuracy
bash scripts/infer_mobilevit_s_fp16_accuracy.sh
# Performance
bash scripts/infer_mobilevit_s_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS     | Top-1(%) | Top-5(%) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| mobilevit_s | 32        | FP16      |1827.75 | 77.127   | 93.546   |
