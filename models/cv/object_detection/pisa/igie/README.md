# PISA (IGIE)

## Model Description

It is a common paradigm in object detection frameworks to treat all samples equally and target at maximizing the performance on average. In this work, we revisit this paradigm through a careful study on how different samples contribute to the overall performance measured in terms of mAP. Our study suggests that the samples in each mini-batch are neither independent nor equally important, and therefore a better classifier on average does not necessarily mean higher mAP. Motivated by this study, we propose the notion of Prime Samples, those that play a key role in driving the detection performance. We further develop a simple yet effective sampling and learning strategy called PrIme Sample Attention (PISA) that directs the focus of the training process towards such samples. Our experiments demonstrate that it is often more effective to focus on prime samples than hard samples when training a detector. Particularly, On the MSCOCO dataset, PISA outperforms the random sampling baseline and hard mining schemes, e.g., OHEM and Focal Loss, consistently by around 2% on both single-stage and two-stage detectors, even with a strong backbone ResNeXt-101.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.12 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth>


Dataset:
  - <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip> to download the labels dataset.
  - <http://images.cocodataset.org/zips/val2017.zip> to download the validation dataset.
  - <http://images.cocodataset.org/zips/train2017.zip> to download the train dataset.

```bash
unzip -q -d ./ coco2017labels.zip
unzip -q -d ./coco/images/ train2017.zip
unzip -q -d ./coco/images/ val2017.zip

coco
├── annotations
│   └── instances_val2017.json
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

```bash
wget https://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- mmcv-*.whl

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# export onnx model
python3 export.py --weight pisa_retinanet_r50_fpn_1x_coco-76409952.pth --cfg pisa_retinanet_r50_fpn_1x_coco.py --output pisa.onnx

# use onnxsim optimize onnx model
onnxsim pisa.onnx pisa_opt.onnx --overwrite-input-shape input:32,3,800,1344
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_pisa_fp16_accuracy.sh
# Performance
bash scripts/infer_pisa_fp16_performance.sh
```

## Model Results

| Model  | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| PISA | 32        | FP16      | 74.109 | 0.513   | 0.333        |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection.git)
