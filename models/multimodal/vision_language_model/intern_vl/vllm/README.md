# InternVL2-4B

## Model Description

InternVL2-4B is a large-scale multimodal model developed by WeTab AI, designed to handle a wide range of tasks involving both text and visual data. With 4 billion parameters, it is capable of understanding and generating complex patterns in data, making it suitable for applications such as image recognition, natural language processing, and multimodal learning.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/OpenGVLab/InternVL2-4B>

```bash
cd ${DeepSparkInference}/models/vision-language-understanding/Intern_VL/vllm
mkdir -p data/intern_vl
ln -s /path/to/InternVL2-4B ./data/intern_vl
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# Contact the iluvatar manager to get adapted install packages of vllm, triton, and ixformer
pip3 install vllm
pip3 install triton
pip3 install ixformer
```

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference_vision_language.py --model ./data/intern_vl/InternVL2-4B --max-tokens 256 -tp 2 --temperature 0.0 --max-model-len 2048
```
