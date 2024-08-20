# FaceNet

## Description

Facenet is a facial recognition system originally proposed and developed by Google. It utilizes deep learning techniques, specifically convolutional neural networks (CNNs), to transform facial images into high-dimensional feature vectors. These feature vectors possess high discriminative power, enabling comparison and identification of different faces. The core idea of Facenet is to map faces into a multi-dimensional space of feature vectors, achieving efficient representation and recognition of faces.

## Setup

### Install

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install tensorflow
pip3 install onnxsim
pip3 install scikit-learn
pip3 install tf_slim
pip3 install tqdm
pip3 install pycuda
pip3 install onnx
pip3 install tabulate
pip3 install cv2
pip3 install scipy==1.8.0
pip3 install pycocotools
pip3 install opencv-python==4.6.0.66
pip3 install simplejson
```

### Download

Pretrained model: <https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz>

Dataset: <https://vis-www.cs.umass.edu/lfw/lfw.tgz> to download the lfw dataset.

```bash
cd ${DeepSparkInference_PATH}/models/cv/face/facenet/ixrt
# download and unzip 20180408-102900.zip
unzip 20180408-102900.zip
```

### Model Conversion

```bash

mkdir -p checkpoints
mkdir -p facenet_weights
git clone https://github.com/timesler/facenet-pytorch
mv /Path/facenet/ixrt/tensorflow2pytorch.py facenet-pytorch
python3 /facenet-pytorch/tensorflow2pytorch.py \
        --facenet_weights_path ./facenet_weights \
        --facenet_pb_path ./20180408-102900 \
        --onnx_save_name facenet_export.onnx
mv facenet_export.onnx ./facenet_weights
```

### Data preprocessing

We need to adjust the image resolution of the original dataset to 160x160. For details, please refer to the following link: <https://blog.csdn.net/rookie_wei/article/details/82078373>. This code relies on tensorflow 1.xx; If you encounter problems with TensorFlow version incompatibility during dataset processing, you can also download the preprocessed dataset from here: <https://github.com/lanrax/Project_dataset/blob/master/facenet_datasets.zip>

```bash
# download and unzip facenet_datasets.zip
wget https://raw.githubusercontent.com/lanrax/Project_dataset/master/facenet_datasets.zip
unzip facenet_datasets.zip
```

## Inference

Because there are differences in model export, it is necessary to verify the following information before executing inference: In deploy.py, "/last_bn/BatchNormalization_output_0" refers to the output name of the BatchNormalization node in the exported ONNX model, such as "1187". "/avgpool_1a/GlobalAveragePool_output_0" refers to the output name of the GlobalAveragePool node, such as "1178". Additionally, make sure to update "/last_bn/BatchNormalization_output_0" in build_engine.py to the corresponding name, such as "1187".

```bash
sed -i -e 's#/last_bn/BatchNormalization_output_0#1187#g' -e 's#/avgpool_1a/GlobalAveragePool_output_0#1178#g' deploy.py build_engine.py
```

### FP16

```bash
# Accuracy
bash scripts/infer_facenet_fp16_accuracy.sh
# Performance
bash scripts/infer_facenet_fp16_performance.sh
```

### INT8

```bash
# Accuracy
bash scripts/infer_facenet_int8_accuracy.sh
# Performance
bash scripts/infer_facenet_int8_performance.sh
```

## Results

| Model   | BatchSize | Precision | FPS       | AUC   | ACC              |
| ------- | --------- | --------- | --------- | ----- | ---------------- |
| FaceNet | 64        | FP16      | 8825.802  | 0.999 | 0.98667+-0.00641 |
| FaceNet | 64        | INT8      | 14274.306 | 0.999 | 0.98633+-0.00605 |
