# BERT Large SQuAD

## Description
BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Setup
### T4 requirement(tensorrt_version >= 8.6)
``` shell
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

### Install 
#### On iluvatar
``` shell
cmake -S . -B build
cmake --build build -j16
```
#### On T4
``` shell
cmake -S . -B build -DUSE_TENSORRT=true
cmake --build build -j16
```

### Download 
``` shell
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

### INT8
```bash
cd python
pip install onnx pycuda
bash script/build_engine --bs 32 --int8
bash script/inference_squad.sh --bs {batch_size} --int8
```

## Results

Model | BatchSize | Precision | FPS | ACC
------|-----------|-----------|-----|----
BERT-Large-SQuAD | 32 | FP16 | Latency QPS: 470.26 sentences/s | "exact_match": 82.36, "f1": 89.68
BERT-Large-SQuAD | 32 | INT8 | Latency QPS: 1490.47 sentences/s | "exact_match": 80.92, "f1": 88.20

## Referenece 
- [bert-large-uncased.zip 外网链接](https://drive.google.com/file/d/1eD8QBkbK6YN-_YXODp3tmpp3cZKlrPTA/view?usp=drive_link)
