# H2OVL Mississippi (vLLM)

## Model Description

The H2OVL-Mississippi-800M is a compact yet powerful vision-language model from H2O.ai, featuring 0.8 billion
parameters. Despite its small size, it delivers state-of-the-art performance in text recognition, excelling in the Text
Recognition segment of OCRBench and outperforming much larger models in this domain. Built upon the robust architecture
of our H2O-Danube language models, the Mississippi-800M extends their capabilities by seamlessly integrating vision and
language tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | vLLM | Release |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.3.0 | >=0.6.4 | 25.09 |
| MR-V100 | 4.2.0 | >=0.6.4 | 25.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/h2oai/h2ovl-mississippi-800m>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "data/Aria"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
python3 offline_inference_vision_language.py --model data/h2ovl-mississippi-800m -tp 1 --max-tokens 256 --trust-remote-code --temperature 0.0 --disable-mm-preprocessor-cache
```

## Model Results
