# MiniCPM V2

## Description

MiniCPM V2 is a compact and efficient language model designed for various natural language processing (NLP) tasks. Building on its predecessor, MiniCPM-V-1, this model integrates advancements in architecture and optimization techniques, making it suitable for deployment in resource-constrained environments.s

## Setup

### Install

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
pip3 install timm==0.9.10
pip3 install transformers
pip3 install --user --upgrade pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Download

- Model: <https://huggingface.co/openbmb/MiniCPM-V-2>
Note: Due to the official weights missing some necessary files for vllm execution, you can download the additional files from here: <https://github.com/HwwwwwwwH/MiniCPM-V-2> to ensure that the file directory matches the structure shown here: <https://github.com/HwwwwwwwH/MiniCPM-V-2>.

```bash
# Download model from the website and make sure the model's path is "data/MiniCPM-V-2"
mkdir data

```

## Inference

```bash
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PATH=/usr/local/corex/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 
```

```bash
wget https://img.zcool.cn/community/012e285a1ea496a8012171323c6bf1.jpg -O dog.jpg
python3 minicpmv-2.0-offline.py --model-path /path/to/model --image-path ./dog.jpg
```
