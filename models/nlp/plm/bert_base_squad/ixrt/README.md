# BERT Base SQuAD (ixRT)

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
mkdir -p data/datasets/bert_base_squad/squad
mkdir -p data/checkpoints/bert_base_squad
wget http://files.deepspark.org.cn:880/deepspark/data/checkpoints/bert_base_uncased_squad.tar
tar -xvf bert_base_uncased_squad.tar -C data/checkpoints/bert_base_squad/
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O data/datasets/bert_base_squad/squad/dev-v1.1.json
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Inference

### FP16

```bash
bash scripts/infer_bert_base_squad_fp16_accuracy.sh
bash scripts/infer_bert_base_squad_fp16_performance.sh
```

## Model Results

| Model           | BatchSize | Precision | Latency QPS | exact_match | f1    |
| --------------- | --------- | --------- | ----------- | ----------- | ----- |
| BERT Base SQuAD | 32        | FP16      | 1444.69     | 80.92       | 88.20 |

## Referenece

- [bert-base-uncased.zip](https://drive.google.com/file/d/1_q7SaiZjwysJ3jWAIQT2Ne-duFdgWivR/view?usp=drive_link)