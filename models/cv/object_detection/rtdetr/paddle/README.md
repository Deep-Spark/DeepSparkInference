# RTDETR (Paddle)

## Model Description

RT-DETR is the first real-time end-to-end object detector. Specifically, we design an efficient hybrid encoder that effectively processes multi-scale features by decoupling intra-scale interaction and cross-scale fusion. Additionally, we propose an IoU-aware query selection mechanism to optimize the initialization of decoder queries. Moreover, RT-DETR supports flexible adjustment of inference speed by using a different number of decoder layers without requiring retraining, which greatly facilitates practical deployment of real-time object detectors.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams>

Dataset:

- <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
- <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
- <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.
- <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> to download the annotations dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip
unzip -q -d ./coco annotations_trainval2017.zip

coco
├── annotations
│   └── instances_train2017.json
│   └── instances_val2017.json
│   └── captions_train2017.json
│   └── captions_val2017.json
│   └── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── images
│   ├── train2017
│   └── val2017
├── labels
│   ├── train2017
│   └── val2017
├── LICENSE
├── README.txt
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt
```

### Install Dependencies

```bash
# Install PaddleDetection
wget https://github.com/PaddlePaddle/PaddleDetection/archive/refs/tags/v2.8.1.zip
unzip v2.8.1.zip
cd PaddleDetection-2.8.1

pip3 install -r requirements.txt
pip3 install -e .
rm -rf v2.8.1.zip
```

### Model Conversion

```bash
# export paddle inference model
python export_model.py -c PaddleDetection-2.8.1/configs/rtdetr/rtdetr_r101vd_6x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams --output_dir=output_inference
```

## Model Inference

```bash
python3 test_det.py \
    --model_path output_inference/rtdetr_r101vd_6x_coco \
    --config PaddleDetection-2.8.1/deploy/auto_compression/configs/rtdetr_reader.yml \
    --precision fp16 \
    --dataset_dir /home/datasets/coco \
    --batch_size 32 \
    --perf_only True
```

### FP16

```bash

# Accuracy
bash scripts/infer_rtdetr_fp16_accuracy.sh
# Performance
bash scripts/infer_rtdetr_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision |  FPS  | IOU@0.5 | IOU@0.5:0.95 |
|:------:|:---------:|:---------:|:-----:|:-------:|:------------:|
| RT-DETR|     32    |   FP16    | 71.4  |  0.729  |     0.543    |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection.git)
