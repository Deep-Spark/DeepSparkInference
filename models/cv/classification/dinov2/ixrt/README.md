# DINOv2-base (ixRT)

## Model Description

DINOv2 is a self-supervised vision Transformer for general-purpose image representation learning.
This example deploys HuggingFace [`onnx-community/dinov2-base`](https://huggingface.co/onnx-community/dinov2-base)
(ONNX export of `facebook/dinov2-base`, ViT-B/14, 86M params) with **ixRT**.

Unlike the IGIE sample (`dinov2/igie`, ViT-S/14 + ImageNet linear probe), this ONNX has **no
classification head** and only exports `last_hidden_state`. Accuracy is measured as:

1. Cosine similarity vs onnxruntime CPU FP32 on the same frozen static graph
2. Downstream leave-one-out 1-NN top-1 on an ImageFolder subset (feature usability)

Retrieval features are built as `CLS ⊕ mean(patch tokens)` → **1536-d**, then L2-normalized.

> **Static shape is required.** The source ONNX uses fully dynamic `pixel_values` and a bicubic
> `Resize` for position-embedding interpolation (trained at 518×518). ixRT builds static engines,
> so `build_engine.py` freezes input to `Bx3xHxW` and constant-folds that `Resize` away before
> engine build.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | dev-only | 26.09 | 

## Directory Layout（非必要章节）

```text
dinov2-ixrt/                    # mirrors models/cv/classification/dinov2/ixrt/
├── README.md
├── build_engine.py             # freeze dynamic ONNX + build ixRT engine
├── inference.py                # perf / acc / knn
├── ixrt_helper.py              # engine IO / plugin helpers
├── serve.py                    # optional HTTP feature service
├── ci/
│   └── prepare.sh              # deps + download model.onnx
└── scripts/
    ├── infer_dinov2_fp16_accuracy.sh
    ├── infer_dinov2_fp16_performance.sh
    └── infer_dinov2_fp16_knn.sh
```

## Model Preparation

### Prepare Resources

Pretrained ONNX: `https://hf-mirror.com/onnx-community/dinov2-base/resolve/main/onnx/model.onnx`

Dataset: any ImageFolder tree (e.g. ImageNet val subset) for `acc` / `knn`. Performance mode
does not need images.

### Install Dependencies

```bash
bash ci/prepare.sh
# or manually:
# pip3 install onnx onnxruntime "numpy<2" cuda-python pillow tqdm
```

Inside CoreX `ml-inference` 4.5.0 images, most deps are already present; only install what is missing.

### Model Conversion

```bash
export MODEL_DIR=/data/models/dinov2-base
export CHECKPOINTS_DIR=${MODEL_DIR}/checkpoints
mkdir -p ${CHECKPOINTS_DIR}

# Freeze input + constant-fold Resize, then build engine (bs=1 latency / bs=32 throughput)
python3 build_engine.py \
    --onnx ${MODEL_DIR}/model.onnx \
    --out-dir ${CHECKPOINTS_DIR} \
    --batch 1 --imgsz 224 --precision fp16

python3 build_engine.py \
    --onnx ${MODEL_DIR}/model.onnx \
    --out-dir ${CHECKPOINTS_DIR} \
    --batch 32 --imgsz 224 --precision fp16

# Optional FP32 engines for precision attribution
python3 build_engine.py \
    --onnx ${MODEL_DIR}/model.onnx \
    --out-dir ${CHECKPOINTS_DIR} \
    --batch 1 --imgsz 224 --precision fp32
```

Artifacts:

```text
checkpoints/
├── dinov2_base_bs1_224.onnx            # static FP32 ONNX (CPU reference)
├── dinov2_base_bs1_224_fp16.engine
├── dinov2_base_bs32_224.onnx
└── dinov2_base_bs32_224_fp16.engine
```

Success check: after freeze, operator list must **not** contain `Resize`, and output shape is
concrete `[B, 257, 768]` for 224 input.

## Model Inference

```bash
export PROJ_DIR=./
export MODEL_DIR=/data/models/dinov2-base
export CHECKPOINTS_DIR=${MODEL_DIR}/checkpoints
export DATASETS_DIR=/path/to/imagenet_val_or_subset
export DEVICE=0   # pick an idle GPU if needed
```

### FP16

```bash
# Accuracy (cosine vs ORT CPU FP32)
bash scripts/infer_dinov2_fp16_accuracy.sh --bs 32 --device ${DEVICE}

# Performance
bash scripts/infer_dinov2_fp16_performance.sh --bs 1 --device ${DEVICE}
bash scripts/infer_dinov2_fp16_performance.sh --bs 32 --device ${DEVICE}

# Downstream kNN (ImageFolder required)
bash scripts/infer_dinov2_fp16_knn.sh --bs 32 --device ${DEVICE}
```

Equivalent direct calls:

```bash
python3 inference.py --mode acc \
  --engine $CHECKPOINTS_DIR/dinov2_base_bs32_224_fp16.engine \
  --onnx   $CHECKPOINTS_DIR/dinov2_base_bs32_224.onnx \
  --images $DATASETS_DIR --limit 128 --device ${DEVICE}

python3 inference.py --mode perf \
  --engine $CHECKPOINTS_DIR/dinov2_base_bs32_224_fp16.engine \
  --warmup 20 --iters 200 --device ${DEVICE}

python3 inference.py --mode knn \
  --engine $CHECKPOINTS_DIR/dinov2_base_bs32_224_fp16.engine \
  --onnx   $CHECKPOINTS_DIR/dinov2_base_bs32_224.onnx \
  --images $DATASETS_DIR --device ${DEVICE}
```

### Optional HTTP service

```bash
pip3 install fastapi uvicorn python-multipart
python3 serve.py --engine $CHECKPOINTS_DIR/dinov2_base_bs1_224_fp16.engine --device ${DEVICE} --port 8100
```

## Model Results

Measured on BI-V150 ×1, CoreX 4.5.0 `ml-inference`, input 224×224.

| Model | Task | BatchSize | Precision | FPS | Metric |
| --------------- | ----------------- | --------- | --------- | ------- | ---------------------- |
| DINOv2-base | Feature extract | 1 | FP16 | 196.29 | latency 5.093 ms |
| DINOv2-base | Feature extract | 32 | FP16 | 401.39 | latency 2.491 ms/img |
| DINOv2-base | Feature extract | 1 | FP32 | 91.36 | latency 10.945 ms |
| DINOv2-base | Feature extract | 32 | FP32 | 176.58 | latency 5.663 ms/img |
| DINOv2-base | vs ORT CPU FP32 | 32 | FP16 | — | embed cos min **0.999972** |
| DINOv2-base | leave-one-out 1-NN | 32 | FP16 | — | top-1 **0.8060** (= ORT) |

> FP16 vs FP32 throughput ≈ **2.2×**. Downstream 1-NN top-1 delta vs ORT is **0.0000** on a
> 500-image / 50-class ImageNet-val subset. Outlier absolute error on raw patch tokens can be
> large under FP16; CLS and aggregated 1536-d retrieval features remain clean — prefer FP32 for
> patch-level downstream tasks.

## References

- [dinov2](https://github.com/facebookresearch/dinov2)
- [onnx-community/dinov2-base](https://huggingface.co/onnx-community/dinov2-base)
- IGIE counterpart: [`models/cv/classification/dinov2/igie`](https://gitee.com/deep-spark/deepsparkinference/tree/master/models/cv/classification/dinov2/igie)
