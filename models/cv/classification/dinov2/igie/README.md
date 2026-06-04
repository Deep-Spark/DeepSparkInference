# DINOv2 ViT-S/14 (IGIE)

## Model Description

DINOv2 is a self-supervised vision Transformer model designed for general-purpose image representation learning. This example uses the ViT-S/14 pretrained backbone for ImageNet linear evaluation: the frozen backbone extracts intermediate features, and a linear classifier is trained and evaluated on top of them.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

visit https://github.com/facebookresearch/dinov2 to see the official repo.

### Prepare Resources

Pretrained model: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`

Dataset: <https://www.image-net.org/download.php> to download the ImageNet-1K dataset.

### Install Dependencies

```bash
git clone https://github.com/facebookresearch/dinov2.git
cp -r eval dinov2/dinov2
cp dinov2-patch/* dinov2/
pip3 install -r requirements.txt requirements-dev.txt
```


### Model Conversion

```bash
cd dinov2
python3 export.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights dinov2_vits14_pretrain.pth \
    --onnx-path dinov2_vits14_pretrain.onnx \
    --input-size 224 \
    --n-last-blocks 4 \
    --device cuda

export batchsize=64
python3 build_engine.py                     \
    --model_path dinov2_vits14_pretrain.onnx              \
    --input input:${batchsize},3,224,224    \
    --precision fp16                        \
    --engine_path dinov2_vits14_pretrain_bs_${batchsize}_fp16.so
```

## Model Inference

```bash
export PYTHONPATH=/path/to/dinov2:${PYTHONPATH}
export IMAGENET_1K=/path/to/ILSVRC2012
export RUN_DIR=../../igie_common/
```

### FP16

```bash
python3 dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights dinov2_vits14_pretrain_bs_64_fp16.so \
    --output-dir ./eval_output/linear_vits14_tvm \
    --ngpus 1 \
    --batch-size 64 \
    --train-dataset ImageNet:split=TRAIN:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra \
    --val-dataset ImageNet:split=VAL:root=${IMAGENET_1K}:extra=${IMAGENET_1K}/extra
```

## Model Results

| Model           | Task        | BatchSize | Precision | FPS | Accuracy |
| --------------- | ----------- | --------- | --------- | --- | -------- |
| DINOv2 ViT-S/14 | Linear Eval | 64        | FP16      |  1132.572   | 81.1%    |
