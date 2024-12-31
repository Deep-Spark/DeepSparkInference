# BERT Large SQuAD

## Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Setup

### Install

```bash
pip3 install -r requirements.txt
```

### Download

Pretrained model: <https://huggingface.co/neuralmagic/bert-large-uncased-finetuned-squadv1>

Dataset: <https://rajpurkar.github.io/SQuAD-explorer>

### Model Conversion

```bash

# Get FP16 Onnx Model
python3 export.py --output bert-large-uncased-squad-v1.onnx

# Do QAT for INT8 test, will take a long time (16 gpus need 1h)
cd Int8QAT

# prepare dataset
mkdir -p data
cp /path/to/SQuAD/train-v1.1.json /path/to/SQuAD/dev-v1.1.json data/

# prepare model into bert-large-uncased from <https://huggingface.co/google-bert/bert-large-uncased/tree/main>
mkdir -p bert-large-uncased

bash run_qat.sh

# model: quant_bert_large/pytorch_model.bin or quant_bert_large/model.safetensors
python3 export_hdf5.py --model quant_bert_large/pytorch_model.bin --model_name large

cd ..

```

## Inference

```bash
export DATASETS_DIR=/Path/to/SQuAD/
```

### FP16

```bash
# Accuracy
bash scripts/infer_bert_large_squad_fp16_accuracy.sh
# Performance
bash scripts/infer_bert_large_squad_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_bert_large_squad_int8_accuracy.sh
# Performance
bash scripts/infer_bert_large_squad_int8_performance.sh
```

## Results

Model            |BatchSize  |SeqLength |Precision |FPS       | F1 Score
-----------------|-----------|----------|----------|----------|--------
Bertlarge(Squad) |    8      |   256    |   FP16   | 302.273  | 91.102
Bertlarge(Squad) |    8      |   256    |   INT8   | 723.169  | 89.899
