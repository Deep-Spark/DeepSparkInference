# LLava

## Model Description

LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture.The LLaVA-NeXT model was proposed in LLaVA-NeXT: Improved reasoning, OCR, and world knowledge by Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee. LLaVa-NeXT (also called LLaVa-1.6) improves upon LLaVa-1.5 by increasing the input image resolution and training on an improved visual instruction tuning dataset to improve OCR and common sense reasoning.

## Model Preparation

### Prepare Resources

-llava-v1.6-vicuna-7b-hf: <https://modelscope.cn/models/swift/llava-v1.6-vicuna-7b-hf>

```bash
# Download model from the website and make sure the model's path is "data/llava"
mkdir data/
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
pip3 install transformers
```

## Model Inference

```bash
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PATH=/usr/local/corex/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 
```

### Inference llava-1.6

```bash
export VLLM_ASSETS_CACHE=../vllm/
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 offline_inference_vision_language.py --model /path/to/model --max-tokens 256 -tp 4 --trust-remote-code --temperature 0.0 --model-type llava-next --max-model-len 4096
```
