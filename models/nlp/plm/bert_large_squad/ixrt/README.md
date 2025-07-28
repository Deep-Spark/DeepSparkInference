# BERT Large SQuAD (ixRT)

## Model Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Get `bert-large-uncased.zip` from [Google
Drive](https://drive.google.com/file/d/1eD8QBkbK6YN-_YXODp3tmpp3cZKlrPTA/view?usp=drive_link)

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- ixrt-1.0.0a0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
- cuda_python-11.8.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

## Model Inference

### FP16

```bash
bash script/infer_bert_large_squad_fp16_accuracy.sh
bash script/infer_bert_large_squad_fp16_performance.sh
```

### INT8

```bash
bash script/infer_bert_large_squad_int8_accuracy.sh
bash script/infer_bert_large_squad_int8_performance.sh
```

| Model              | BatchSize   | Precision   | Latency QPS           | exact_match   | f1      |
| :----: | :----: | :----: | :----: | :----: | :----: |
| BERT-Large-SQuAD   | 32          | FP16        | 470.26 sentences/s    | 82.36         | 89.68   |
| BERT-Large-SQuAD   | 32          | INT8        | 1490.47 sentences/s   | 80.92         | 88.20   |
| ------------------ | ----------- | ----------- | --------------------- | ------------- | ------- |
| BERT-Large-SQuAD   | 32          | FP16        | 470.26 sentences/s    | 82.36         | 89.68   |
| BERT-Large-SQuAD   | 32          | INT8        | 1490.47 sentences/s   | 80.92         | 88.20   |
