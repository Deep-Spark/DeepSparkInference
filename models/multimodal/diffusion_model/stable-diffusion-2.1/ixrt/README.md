# Stable Diffusion 2.1

## Model description

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

Pretrained model: <http://files.deepspark.org.cn:880/deepspark/data/checkpoints/stable_diffusion_2_1/>

Datasets: <http://files.deepspark.org.cn:880/deepspark/data/datasets/stable_diffusion_2_1/>

### Prepare Resources

```bash
mkdir -p checkpoints/stable_diffusion_2_1_ixrt
# download all files into checkpoints/stable_diffusion_2_1_ixrt
mkdir -p datasets/stable_diffusion_2_1_ixrt
# download all files into datasets/stable_diffusion_2_1_ixrt and unzip tokenizer.zip
```

### Install Dependencies

```bash
pip3 install torch transformers diffusers onnx Pillow numpy scikit-image opencv-python==4.5.5.64

git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git

export OSS_ENV=iluvatar-corex-ixrt
```

## Model Inference

```bash
bash infer_fp16_stable_diffusion_2_1.sh
```

## Model Results

|   Module      |   Latency    |
| :----: | :----: |
|      CLIP       |      9.12 ms |
|    UNet x 20    |    832.62 ms |
|     VAE-Dec     |     54.80 ms |
|-----------------|--------------|
|    Pipeline     |    897.32 ms |
|-----------------|--------------|

```
图像比较结果:
不同像素数量: 688,191 / 786,432
差异百分比: 87.508011%
平均像素差异: 3.5578
结构相似性(SSIM): 0.9481
直方图相关性: 0.9988
```