# BERT Large SQuAD (ixRT)

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

```bash
mkdir -p data/datasets
mkdir -p data/checkpoints
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/bert-large-uncased.tar
tar -xvf bert-large-uncased.tar -C data/checkpoints
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O data/datasets/dev-v1.1.json
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- ixrt-*.whl
- cuda_python-*.whl

## Model Inference

### FP16

```bash
bash scripts/infer_bert_large_squad_fp16_accuracy.sh
bash scripts/infer_bert_large_squad_fp16_performance.sh
```

### INT8

```bash
bash scripts/infer_bert_large_squad_int8_accuracy.sh
bash scripts/infer_bert_large_squad_int8_performance.sh
```

## Model Results

| Model              | BatchSize   | Precision   | Latency QPS           | exact_match   | f1      |
| :----: | :----: | :----: | :----: | :----: | :----: |
| BERT-Large-SQuAD   | 32          | FP16        | 470.26 sentences/s    | 82.36         | 89.68   |
| BERT-Large-SQuAD   | 32          | INT8        | 1490.47 sentences/s   | 80.92         | 88.20   |