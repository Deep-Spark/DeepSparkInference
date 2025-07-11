# BERT Base NER (IGIE)

## Model Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/bert-base-chinese>

Dataset: <http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz>

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
export DATASETS_DIR=/Path/to/china-people-daily-ner-corpus/

# Get pytorch weights
python3 get_weights.py

# Do QAT for INT8 test, will take a long time  
cd Int8QAT/
python3 run_qat.py --model_dir ../test/ --datasets_dir ${DATASETS_DIR}
python3 export_hdf5.py --model quant_base/pytorch_model.bin
cd ../
```

## Model Inference

### INT8

```bash
# Accuracy
bash scripts/infer_bert_base_ner_int8_accuracy.sh
# Performance
bash scripts/infer_bert_base_ner_int8_performance.sh
```

## Model Results

| Model         | BatchSize | SeqLength | Precision | FPS      | F1 Score |
| :----: | :----: | :----: | :----: | :----: | :----: |
| BERT Base NER | 8         | 256       | INT8      | 2067.252 | 96.2     |
