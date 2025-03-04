# BERT Large SQuAD

## Model Description

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Model Preparation

Get `bert-large-uncased.zip` from [Google
Drive](https://drive.google.com/file/d/1eD8QBkbK6YN-_YXODp3tmpp3cZKlrPTA/view?usp=drive_link)

### NV requirement(tensorrt_version >= 8.6)

```bash
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

## Install

```bash
pip3 install -r requirements.txt
```

### On Iluvatar

```bash
cmake -S . -B build
cmake --build build -j16
```

### On NV

```bash
cmake -S . -B build -DUSE_TENSORRT=true
cmake --build build -j16
```

## Download

```bash
cd python
bash script/prepare.sh v1_1
```

## Model Inference

### FP16

```bash
cd python

# use --bs to set max_batch_size (dynamic)
bash script/build_engine.sh --bs 32
bash script/inference_squad.sh --bs 32
```

### INT8

```bash
cd python
pip install onnx pycuda
bash script/build_engine.sh --bs 32 --int8
bash script/inference_squad.sh --bs 32 --int8
```
| Model            | BatchSize | Precision | Latency QPS         | exact_match | f1    |
|------------------|-----------|-----------|---------------------|-------------|-------|
| BERT-Large-SQuAD | 32        | FP16      | 470.26 sentences/s  | 82.36       | 89.68 |
| BERT-Large-SQuAD | 32        | INT8      | 1490.47 sentences/s | 80.92       | 88.20 |
|------------------|-----------|-----------|---------------------|-------------|-------|
| BERT-Large-SQuAD | 32        | FP16      | 470.26 sentences/s  | 82.36       | 89.68 |
| BERT-Large-SQuAD | 32        | INT8      | 1490.47 sentences/s | 80.92       | 88.20 |
