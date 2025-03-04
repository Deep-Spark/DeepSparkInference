# Qwen-7B (vLLM)

## Model Description

Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language
processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first
installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct
models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat
models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance
across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning
from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities
for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks
like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and
Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These
models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind
the proprietary models.

## Model Preparation

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource
center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

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

### Prepare Resources

- Model: - Model: <https://modelscope.cn/models/qwen/Qwen-7B/summary>

```bash
cd ${DeepSparkInference}/models/nlp/large_language_model/qwen-7b/vllm
mkdir -p data/qwen
ln -s /path/to/Qwen-7B ./data/qwen
```

## Model Inference

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 offline_inference.py --model ./data/qwen/Qwen-7B-Chat --max-tokens 256 -tp 2 --trust-remote-code --temperature 0.0
```
