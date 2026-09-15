# ArcFace

## Model Description

ArcFace is a face recognition method that maps faces to a hypersphere and adds an angular margin to the target class, producing highly discriminative embeddings. This workspace runs the official InsightFace [arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) IResNet-50 backbone trained on MS1MV3.

## Supported Environments

| GPU| [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model (official `arcface_torch` MS1MV3 IResNet50, non-commercial research only):

- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)
- [Baidu Yun](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g) (code: `e8pw`)

Download `arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth` and place it in this directory as `backbone.pth`.

Dataset: InsightFace aligned LFW verification pack (`lfw.bin`). Preferred source is the MS1MV3 release:

- [GDrive](https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view?usp=sharing)
- [Baidu](https://pan.baidu.com/s/1RBnaW88PC6cKqtYwgfVX8Q) (code: `8eb3`)

```bash
ls ${DATASETS_DIR}/lfw.bin
```

```text
${DATASETS_DIR}
└── lfw.bin
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
```

### Model Conversion

```bash
# download backbone.pth from the recommend link and place it here
python3 export.py --weight backbone.pth --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/insightface_val_bins/
```

### FP16

```bash
# Accuracy
bash scripts/infer_arcface_fp16_accuracy.sh
# Performance
bash scripts/infer_arcface_fp16_performance.sh
```

## Model Results

Official InsightFace MS1MV3 IResNet50 reference on LFW is **99.80%** (model zoo). FPS below is left blank until measured on MR-V100.

| Model   | BatchSize | Precision |    FPS    | LFW ACC |
| :-----: | :-------: | :-------: | :-------: | :-----: |
| ArcFace | 32        | FP16      | 2866.02   | 0.9980  |

## References

- [deepinsight/insightface](https://github.com/deepinsight/insightface)
- [arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
