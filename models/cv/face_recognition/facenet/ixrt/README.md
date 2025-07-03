# FaceNet (IxRT)

## Model Description

Facenet is a facial recognition system originally proposed and developed by Google. It utilizes deep learning techniques, specifically convolutional neural networks (CNNs), to transform facial images into high-dimensional feature vectors. These feature vectors possess high discriminative power, enabling comparison and identification of different faces. The core idea of Facenet is to map faces into a multi-dimensional space of feature vectors, achieving efficient representation and recognition of faces.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz>

Dataset: <https://vis-www.cs.umass.edu/lfw/lfw.tgz> to download the lfw dataset.

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
mkdir -p checkpoints
mkdir -p facenet_weights
cd facenet_weights
wget http://files.deepspark.org.cn:880/deepspark/facenet_export.onnx
```

### Data preprocessing

We need to adjust the image resolution of the original dataset to 160x160. For details, please refer to the following link: <https://blog.csdn.net/rookie_wei/article/details/82078373>. This code relies on tensorflow 1.xx; If you encounter problems with TensorFlow version incompatibility during dataset processing, you can also download the preprocessed dataset from here: <https://github.com/lanrax/Project_dataset/blob/master/facenet_datasets.zip>

```bash
# download and unzip facenet_datasets.zip
wget https://raw.githubusercontent.com/lanrax/Project_dataset/master/facenet_datasets.zip
unzip facenet_datasets.zip
```

## Model Inference

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

## Model Results

| Model   | BatchSize | Precision | FPS       | AUC   | ACC              |
| ------- | --------- | --------- | --------- | ----- | ---------------- |
| FaceNet | 64        | FP16      | 8825.802  | 0.999 | 0.98667+-0.00641 |
| FaceNet | 64        | INT8      | 14274.306 | 0.999 | 0.98633+-0.00605 |
