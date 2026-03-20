# Step3 (vLLM)

## Model Description

Step3 is cutting-edge multimodal reasoning model—built on a Mixture-of-Experts architecture with 321B total parameters and 38B active. It is designed end-to-end to minimize decoding costs while delivering top-tier performance in vision–language reasoning. Through the co-design of Multi-Matrix Factorization Attention (MFA) and Attention-FFN Disaggregation (AFD), Step3 maintains exceptional efficiency across both flagship and low-end accelerators.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | dev-only | 25.12 |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://www.modelscope.cn/models/stepfun-ai/step3>

```bash
# Download model from the website and make sure the model's path is "data/step3"
mkdir data
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/MMMU_BETA.json
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/MMSTAR_BETA.json
pip3 install -r requirements.txt
```

## Model Inference

### Inference with W4A8

#### Performance Test

1. Set environment variables:
```bash
export VLLM_W8A8_MOE_USE_W4A8=1
export VLLM_ENFORCE_CUDA_GRAPH=1
```

2. Start server:
```bash
vllm serve /path/to/model --limit-mm-per-prompt '{"image":5}'  --gpu-memory-utilization 0.92 --port 12347 --trust-remote-code --disable-cascade-attn  --no-enable-prefix-caching  --max-model-len 65536   --tensor-parallel-size 4 --pipeline-parallel-size 4  --max-num-seqs 1024  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

3. Run client (Input1024, Output1024, BS10):
```bash
vllm bench serve --num-prompts 4*[max-concurrency] --model /path/to/model --dataset-name random --random-input-len 1024 --random-output-len 1024 --max-concurrency 10 --host 0.0.0.0 --port 12347  --disable-tqdm --ignore-eos
```

#### Accuracy Test

4. The evaluation scripts are already included in this directory:
```bash
# eval_dataset.py and eval_dataset_w8a8.py are in the current directory
pip install fire
```

5. Set environment variables:
```bash
export VLLM_W8A8_MOE_USE_W4A8=1
export VLLM_ENFORCE_CUDA_GRAPH=1
```

6. Start server:
```bash
vllm serve /path/to/model --limit-mm-per-prompt '{"image":5}'  --gpu-memory-utilization 0.92 --port 12347 --trust-remote-code --disable-cascade-attn  --no-enable-prefix-caching  --max-model-len 65536   --tensor-parallel-size 4 --pipeline-parallel-size 4  --max-num-seqs 1024  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

7. Run client (MMMU dataset):
```bash
pip install fire
python3 eval_dataset.py --dataset_name MMMU_BETA --model /path/to/model  --ip 127.0.0.1 --port 12347 --num_workers 8
```

### Inference with w8a8
#### Starting w8a8 server
```bash
python3 bf162int8.py --input-bf16-hf-path data/step3 --output-int8-hf-path data/step3_w8a8/

# starting server
VLLM_DEFAULT_HIGH_RESOLUTION=true VLLM_W8A8_MOE_USE_W4A8=0 VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server \
--model data/step3_w8a8/   --max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.92 --port 12347 \
--trust-remote-code \
--disable-cascade-attn --no-enable-prefix-caching  \
--max-model-len 30720  --seed 42  -tp 8 -pp 2 -dp 1 --max-num-seqs 4 --limit-mm-per-prompt image=5
```
#### Testing
```bash
curl 127.0.0.1:12347/v1/completions -H "Content-Type: application/json" -d '{"model":"data/step3_w8a8/",
"prompt":"简单介绍一下上海?",
"temperature":0.0,
"max_tokens":128}'

# acc test
python3 eval_dataset_w8a8.py --dataset_name MMSTAR_BETA --model data/step3_w8a8/  --ip 127.0.0.1 --port 12347 --num_workers 4 
```

### Inference with w4a8
#### Starting w4a8 server
```bash
python3 bf16Toint4.py --input-fp8-hf-path data/step3/ --output-int8-hf-path data/step3_w4a8_TN/ --group-size -1 --format TN --version 2 

VLLM_DEFAULT_HIGH_RESOLUTION=true VLLM_W8A8_MOE_USE_W4A8=1 VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server \
--model data/step3_w4a8_TN/   --max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.92 --port 12347 \
--trust-remote-code \
--disable-cascade-attn  --no-enable-prefix-caching \
--max-model-len 61440  --seed 42  -tp 4 -pp 4 -dp 1 --max-num-seqs 16 --limit-mm-per-prompt image=5
```

#### Testing
```bash
curl 127.0.0.1:12347/v1/completions -H "Content-Type: application/json" -d '{"model":"data/step3_w4a8_TN/",
"prompt":"简单介绍一下上海?",
"temperature":0.0,
"max_tokens":128}'

# acc test
python3 eval_dataset.py --dataset_name MMMU_BETA --model data/step3_w4a8_TN/  --ip 127.0.0.1 --port 12347 --num_workers 16
```

## Model Results
|Model|MMSTAR_BETA|MMMU_BETA|
|:---:|:---:|:---:|
|step3_w8a8|0.710|0.745|
|step3_w4a8|0.705|0.730|
