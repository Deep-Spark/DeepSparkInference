# YOLOv8n-Face

## Model Description

YOLOv8n-Face is a real-time face detector with 5-point landmarks, based on [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face). It extends Ultralytics YOLOv8 Pose and is evaluated on WIDER FACE.

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 5.0.0 | 26.09 |

## Model Preparation

### Prepare Resources

Pretrained model (`yolov8n-face.pt`): [Google Drive](https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing)

Place the weight in this directory as `yolov8n-face.pt`.

Dataset: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) validation images. List file `wider_val.txt`

```text
${DATASETS_DIR}
└── val
    ├── wider_val.txt
    └── images
        ├── 0--Parade
        └── ...
```

The Easy / Medium / Hard `.mat` files are **not** in that repo; they come from the official WIDER FACE MATLAB eval package ([WIDERFACE](http://shuoyang1213.me/WIDERFACE/)).

Accuracy evaluation uses the open-source Python toolkit [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation).

```bash
git clone https://github.com/wondervictor/WiderFace-Evaluation.git widerface_evaluate

# official eval_tools (contains ground_truth/*.mat)
wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
unzip eval_tools.zip
mkdir -p ground_truth
cp eval_tools/ground_truth/wider_*.mat ground_truth/
```

Required files:

```text
ground_truth/
├── wider_face_val.mat
├── wider_easy_val.mat
├── wider_medium_val.mat
└── wider_hard_val.mat
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Model Conversion

```bash
# download yolov8n-face.pt from the recommend link and place it here
python3 export.py --weight yolov8n-face.pt --batch 32
```

`export.py` wraps `torch.onnx.export` with `dynamo=False` before `YOLO.export()`, because PyTorch 2.9+ defaults Dynamo ONNX which IGIE does not support, and Ultralytics does not always pass this flag.

## Model Inference

```bash
export DATASETS_DIR=/Path/to/widerface/
```

### FP16

```bash
# Accuracy
bash scripts/infer_yolov8n_face_fp16_accuracy.sh
# Performance
bash scripts/infer_yolov8n_face_fp16_performance.sh
```

## Model Results

| Model        | BatchSize | Precision     |    FPS   | Easy AP(%) | Medium AP(%) | Hard AP(%) |
| :----------: | :-------: | :-----------: | :------: | :--------: | :----------: | :--------: |
| YOLOv8n-Face | 32        |     FP16      | 2111.245 | 94.46      | 92.18        | 79.02      |

## References

- [derronqi/yolov8-face](https://github.com/derronqi/yolov8-face)
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [wondervictor/WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation)
