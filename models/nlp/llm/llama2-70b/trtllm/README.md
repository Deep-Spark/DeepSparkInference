# LlaMa2 70B (TensorRT-LLM)

## Model Description

we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale
from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.
Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for
helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our
approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work
and contribute to the responsible development of LLMs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/meta-llama/Llama-2-70b-chat-hf>

- Dataset:<https://huggingface.co/datasets/cnn_dailymail>

```bash
# Download model from the website and make sure the model's path is "data/llama2-70b-chat"
# Download dataset from the website and make sure the dataset's path is "data/datasets_cnn_dailymail"
mkdir data

# Please download rouge.py to this path if your server can't attach huggingface.co.
mkdir -p rouge/
wget --no-check-certificate https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/rouge/rouge.py -P rouge
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

bash scripts/set_environment.sh .
```

## Model Inference

### FP16

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Build engine
bash scripts/test_trtllm_llama2_70b_gpu8_build.sh
# Inference
bash scripts/test_trtllm_llama2_70b_gpu8.sh
```
