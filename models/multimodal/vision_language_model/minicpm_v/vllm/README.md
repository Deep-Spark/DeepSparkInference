# MiniCPM-V 2 (vLLM)

## Model Description

MiniCPM V2 is a compact and efficient language model designed for various natural language processing (NLP) tasks.
Building on its predecessor, MiniCPM-V-1, this model integrates advancements in architecture and optimization
techniques, making it suitable for deployment in resource-constrained environments.s

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/openbmb/MiniCPM-V-2_6>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "./minicpm_v"
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip install timm==0.9.10
```

## Model Inference

```bash
python3 offline_inference_vision_language.py --model-type minicpmv
```

## Model Results
