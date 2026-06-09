# Mask R-CNN (ixRT)

## Model Description

Mask R-CNN (Mask Region-Based Convolutional Neural Network) is an extension of the Faster R-CNN model, which is itself an improvement over R-CNN and Fast R-CNN. Developed by Kaiming He et al., Mask R-CNN is designed for object instance segmentation tasks, meaning it not only detects objects within an image but also generates high-quality segmentation masks for each instance.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.2.0 | 25.03 | release/25.03 |

**This model is compatible with IXUCA SDK up to version 4.2.0.**

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/25.03`

## Model Preparation

### Prepare Resources

Prepare weights and datasets referring to below steps:

"maskrcnn.wts" [export method](https://github.com/wang-xinyu/tensorrtx/tree/master/rcnn#how-to-run)

- use the [script](https://github.com/wang-xinyu/tensorrtx/blob/master/rcnn/gen_wts.py)
- use the [config file](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml)
- use [weights](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl)

```bash
# put maskrcnn.wts in "python/maskrcnn.wts"
wget -P python/ http://files.deepspark.org.cn:880/deepspark/wts/maskrcnn.wts
```

Visit [COCO site](https://cocodataset.org/) and get COCO2017 datasets

- images directory: coco/images/val2017/*.jpg
- annotations directory: coco/annotations/instances_val2017.json

### Install Dependencies

Prepare on MR GPU

```bash
cd scripts/
bash init.sh
```

Prepare on NV GPU

```bash
cd scripts/
bash init_nv.sh
```

## Model Inference

### FP16

```bash
cd ../
# Performance
bash scripts/infer_mask_rcnn_fp16_performance.sh
# Accuracy
bash scripts/infer_mask_rcnn_fp16_accuracy.sh
```

## Model Results

| Model      | BatchSize | Precision | FPS   | ACC                                            |
| :----: | :----: | :----: | :----: | :----: |
| Mask R-CNN | 1         | FP16      | 12.15 | bbox mAP@0.5 :  0.5512, segm mAP@0.5 :  0.5189 |

## Refereneces

- [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/rcnn)
- [detectron2](https://github.com/facebookresearch/detectron2)
