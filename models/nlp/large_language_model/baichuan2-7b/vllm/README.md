# Baichuan-2-7B 

## Description
Baichuan 2 is a new generation open-source large language model launched by Baichuan Intelligence. It is trained on high-quality data with 26 trillion tokens, which sounds like a substantial dataset. Baichuan 2 achieves state-of-the-art performance on various authoritative Chinese, multilingual, and domain-specific benchmarks of similar size, indicating its excellent capabilities in language understanding and generation.This release includes Base and Chat versions of 7B. 

## Setup

### Install
In order to run the model smoothly, we need the following dependency files:
1. ixrt-xxx.whl
2. ixformer-xxx.whl
3. vllm-xxx.whl
Please contact the staff to obtain the relevant installation packages.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install transformers==4.33.2
pip3 install Path/To/ixrt-xxx.whl
pip3 install Path/To/vllm-xxx.whl
pip3 install Path/To/ixformer-xxx.whl
```

### Download
Pretrained model: <https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main>

```bash
mkdir /data/baichuan/
mv Baichuan2-7B-Base.tar/zip /data/baichuan/
```


## Run model

```bash
python3 offline_inference.py --model /data/baichuan/Baichuan2-7B-Base/ --chat_template template_baichuan.jinja --trust-remote-code
```

## Run Baichuan w8a16 quantization

### Retrieve int8 weights

Int8 weights will be saved at /data/baichuan/Baichuan2-7B-Base/int8
```bash
python3 convert2int8.py --model-path /data/baichuan/Baichuan2-7B-Base/
```

### Run

```bash
python3 offline_inference.py --model /data/baichuan/Baichuan2-7B-Base/int8/ --chat_template template_baichuan.jinja --quantization w8a16 --trust-remote-code --max-num-seqs 1 --max-model-len 256 \    
                             --trust-remote-code --tensor-parallel-size 2 --temperature 0.0
```