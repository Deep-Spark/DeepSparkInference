# BERT Base SQuAD

## Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Setup

### T4 requirement(tensorrt_version >= 8.6)

```bash
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

## Install

```bash
pip install onnx
pip install pycuda
```

### Install on Iluvatar

```bash
cmake -S . -B build
cmake --build build -j16
```

### Install on T4

```bash
cmake -S . -B build -DUSE_TENSORRT=true
cmake --build build -j16
```

## Download

```bash
cd python
bash script/prepare.sh v1_1
```

## Inference

### On Iluvatar

#### FP16

```bash
cd script

# FP16
bash infer_bert_base_squad_fp16_ixrt.sh

# INT8
bash infer_bert_base_squad_int8_ixrt.sh
```

### On NV

```bash
# FP16
# use --bs to set max_batch_size (dynamic) 
bash script/build_engine.sh --bs 32
bash script/inference_squad.sh --bs 32

# INT8
bash script/build_engine.sh --bs 32 --int8
bash script/inference_squad.sh --bs 32 --int8
```

## Results

| Model           | BatchSize | Precision | Latency QPS | exact_match | f1    |
| --------------- | --------- | --------- | ----------- | ----------- | ----- |
| BERT Base SQuAD | 32        | FP16      | 1444.69     | 80.92       | 88.20 |
| BERT Base SQuAD | 32        | INT8      | 2325.20     | 78.41       | 86.97 |

## Referenece

- [bert-base-uncased.zip](https://drive.google.com/file/d/1_DJDdKBanqJ6h3VGhH78F9EPgE2wK_Tw/view?usp=drive_link)
