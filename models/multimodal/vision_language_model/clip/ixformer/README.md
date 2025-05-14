# CLIP (IxFormer)

## Model Description

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet zero-shot without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

Pretrained model: Go to the website <https://huggingface.co/models> to find the pre-trained model you need. Here, we choose clip-vit-base-patch32.

```bash
# Download model from the website and make sure the model's path is "data/clip-vit-base-patch32"
mkdir -p data
unzip clip-vit-base-patch32.zip -d data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -U transformers==4.27.1
```

## Model Inference

### Test using the OpenAI interface

Please modify the part in the test_clip.py script that pertains to the model path.

```bash
python3 inference.py
```
