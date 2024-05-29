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

### FP16

```bash
cd python
pip install onnx pycuda
# use --bs to set max_batch_size (dynamic) 
bash script/build_engine --bs 32
bash script/inference_squad.sh --bs {batch_size}
```

## Results

Model | BatchSize | Precision | FPS | ACC
------|-----------|-----------|-----|----
BERT-Base-SQuAD | 32 | fp16 | Latency QPS: 1543.40 sentences/s | "exact_match": 80.92, "f1": 88.20
