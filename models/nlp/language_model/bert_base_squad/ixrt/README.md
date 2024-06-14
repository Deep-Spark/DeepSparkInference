# BERT Base SQuAD

## Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Setup

### T4 requirement(tensorrt_version >= 8.6)

```bash
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

### Install

#### On iluvatar

```bash
cmake -S . -B build
cmake --build build -j16
```

#### On T4

```bash
cmake -S . -B build -DUSE_TENSORRT=true
cmake --build build -j16
```

### Download

```bash
cd python
bash script/prepare.sh v1_1
```

## Inference

### On T4

```bash
# FP16
cd python
pip install onnx pycuda
# use --bs to set max_batch_size (dynamic) 
bash script/build_engine.sh --bs 32
bash script/inference_squad.sh --bs 32
```

```bash
# INT8
cd python
pip install onnx pycuda
bash script/build_engine.sh --bs 32 --int8
bash script/inference_squad.sh --bs 32 --int8
```
#### On iluvatar

```bash
# FP16
cd python/script
bash infer_bert_base_squad_fp16_ixrt.sh
```

```bash
# INT8
cd python/script
bash infer_bert_base_squad_int8_ixrt.sh
```

## Results

Model | BatchSize | Precision | FPS | ACC
------|-----------|-----------|-----|----
BERT-Base-SQuAD | 32 | fp16 | Latency QPS: 1543.40 sentences/s | "exact_match": 80.92, "f1": 88.20

## Referenece 
- [bert-base-uncased.zip 外网链接](https://drive.google.com/file/d/1_DJDdKBanqJ6h3VGhH78F9EPgE2wK_Tw/view?usp=drive_link)
