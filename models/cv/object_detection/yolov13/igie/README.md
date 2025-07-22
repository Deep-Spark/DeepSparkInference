# YOLOv13 (IGIE)

## Model Description

YOLOv13 addresses the detection performance bottlenecks of the traditional YOLO series in complex scenarios through innovative HyperACE and FullPAD mechanisms. Additionally, it incorporates lightweight design to significantly reduce computational complexity and parameter count, making it an accurate and efficient object detection model.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt>

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
pip3 install -r requirements.txt
```

## Model Conversion

```bash
git clone --depth 1 https://github.com/iMoonLab/yolov13.git
cd yolov13
pip3 install -e .
cd ..

python3 export.py --weight yolov13n.pt --batch 32
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov13_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov13_fp16_performance.sh
```

## Model Results

| Model   | BatchSize | Precision | FPS     | IOU@0.5 | IOU@0.5:0.95 |
| ------- | --------- | --------- | ------- | ------- | ------------ |
| YOLOv13 | 32        | FP16      | 400.68  | 0.574   | 0.412        |

## References

- [YOLOv13](https://github.com/iMoonLab/yolov13)
