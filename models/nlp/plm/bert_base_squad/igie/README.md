# BERT Base SQuAD (IGIE)

## Model Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

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

Pretrained model: <https://huggingface.co/csarron/bert-base-uncased-squad-v1>

Dataset: <https://rajpurkar.github.io/SQuAD-explorer>

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
python3 export.py --output bert-base-uncased-squad-v1.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/SQuAD/
```

### FP16

```bash
# Accuracy
bash scripts/infer_bert_base_squad_fp16_accuracy.sh
# Performance
bash scripts/infer_bert_base_squad_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | SeqLength | Precision | FPS    | F1 Score |
| --------------- | --------- | --------- | --------- | ------ | -------- |
| BERT Base SQuAD | 8         | 256       | FP16      | 901.81 | 88.08    |
