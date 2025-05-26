# Idefics3 (vLLM)

## Model Description

Idefics3 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text
outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple
images, or simply behave as a pure language model without visual inputs. It improves upon Idefics1 and Idefics2,
significantly enhancing capabilities around OCR, document understanding and visual reasoning.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | vLLM | Release |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.2.0 | >=0.6.4 | 25.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3>

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
python3 offline_inference_vision_language.py --model data/Idefics3-8B-Llama3 -tp 4 --max-tokens 256 --trust-remote-code --temperature 0.0 --disable-mm-preprocessor-cache
```

## Model Results
