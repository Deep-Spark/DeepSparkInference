# DenseNet201

## Model Description

DenseNet201 is a deep convolutional neural network that stands out for its unique dense connection architecture, where each layer integrates features from all previous layers, effectively reusing features and reducing the number of parameters. This design not only enhances the network's information flow and parameter efficiency but also increases the model's regularization effect, helping to prevent overfitting. DenseNet201 consists of multiple dense blocks and transition layers, capable of capturing rich feature representations while maintaining computational efficiency, making it suitable for complex image recognition tasks.

## Model Preparation

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install tabulate
pip3 install ppq
pip3 install tqdm
pip3 install cuda-python
```

### Prepare Resources

Pretrained model: <https://download.pytorch.org/models/densenet201-c1103571.pth>

Dataset: <https://www.image-net.org/download.php> to download the validation dataset.

### Model Conversion

```bash
python3 export.py --weight densenet201-c1103571.pth --output densenet201.onnx
```

## Model Inference

```bash
export DATASETS_DIR=/Path/to/imagenet_val/
```

### FP16

```bash
# Accuracy
bash scripts/infer_densenet201_fp16_accuracy.sh
# Performance
bash scripts/infer_densenet201_fp16_performance.sh
```

## Model Results

| Model       | BatchSize | Precision | FPS      | Top-1(%) | Top-5(%) |
| ----------- | --------- | --------- | -------- | -------- | -------- |
| DenseNet201 | 32        | FP16      | 788.946  | 76.88    | 93.37    |
