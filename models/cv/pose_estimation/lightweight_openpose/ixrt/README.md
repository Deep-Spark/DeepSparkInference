# Lightweight OpenPose (ixRT)

## Model Description

This work heavily optimizes the OpenPose approach to reach real-time inference on CPU with negliable accuracy drop. It
detects a skeleton (which consists of keypoints and connections between them) to identify human poses for every person
inside the image. The pose may contain up to 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips,
knees, and ankles. On COCO 2017 Keypoint Detection validation set this code achives 40% AP for the single scale
inference (no flip or any post-processing done).

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

Pretrained model: <https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth>

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
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# export onnx model
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
cd lightweight-human-pose-estimation.pytorch
mv scripts/convert_to_onnx.py .
python3 convert_to_onnx.py --checkpoint-path /Path/to/checkpoint_iter_370000.pth
cd ..
mkdir -p checkpoints
onnxsim ./lightweight-human-pose-estimation.pytorch/human-pose-estimation.onnx ./checkpoints/lightweight_openpose.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/coco/
export CHECKPOINTS_DIR=/Path/to/checkpoints/
```

### FP16

```bash
# Accuracy
bash scripts/infer_lightweight_openpose_fp16_accuracy.sh
# Performance
bash scripts/infer_lightweight_openpose_fp16_performance.sh
```

## Model Results

| Model                | BatchSize | Precision | FPS       | IOU@0.5 | IOU@0.5:0.95 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Lightweight OpenPose | 1         | FP16      | 21030.833 | 0.660   | 0.401        |

## References

- [lightweight-human-pose-estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
