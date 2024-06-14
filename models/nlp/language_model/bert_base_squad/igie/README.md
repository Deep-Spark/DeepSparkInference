# BERT Base SQuAD

## Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Setup

### Install

```bash
pip3 install onnx
pip3 install tqdm
pip3 install transformers
```

### Download

Pretrained model: <https://huggingface.co/csarron/bert-base-uncased-squad-v1>

Dataset: <https://rajpurkar.github.io/SQuAD-explorer>

### Model Conversion

```bash
python3 export.py --output bert-base-uncased-squad-v1.onnx
```

## Inference

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

## Results

Model            |BatchSize  |SeqLength |Precision |FPS       | F1 Score
-----------------|-----------|----------|----------|----------|--------
Bertbase(Squad)  |    8      |   256    |   FP16   |901.81    | 88.08
