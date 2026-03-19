# Qwen2.5-VL (vLLM)

## Model Description

Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images. Qwen2.5-VL directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use. Qwen2.5-VL can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments. Qwen2.5-VL can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes. for data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |
| MR-V100 | 4.4.0     |  26.03  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
export VLLM_ASSETS_CACHE=../vllm/
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
python3 offline_inference_vision_language.py --model /path/to/Qwen2.5-VL-3B-Instruct/ -tp 4 --trust-remote-code --temperature 0.0 --max-token 256
```

### Qwen2.5-VL-32B-Instruct (W8A8/W4A16)

#### Performance Test

1. Set environment variables:
```bash
export VLLM_ENFORCE_CUDA_GRAPH=1
```

2. Start server:
```bash
vllm serve /path/to/model  --max-num-seqs 1 --max-model-len 98304 --limit_mm_per_prompt '{"image": 5}' --disable-cascade-attn --tensor-parallel-size 4 --gpu_memory_utilization 0.9 --pipeline-parallel-size 1 --host 0.0.0.0 --port 8000  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

3. Run client:
```bash
# Use the pre-copied guidellm
cd guidellm && pip install .
pip install beautifulsoup4
cd ..
guidellm --data "prompt_tokens=512,generated_tokens=512,images=1,width=1770,height=1180" --data-type emulated --model /path/to/model --target "http://localhost:8000/v1" --max-requests 1
```

### Qwen2.5-VL-72B-Instruct (W4A16)

#### Performance Test

1. Set environment variables:
```bash
export VLLM_ENFORCE_CUDA_GRAPH=1
```

2. Start server:
```bash
vllm serve /path/to/model  --max-num-seqs 1 --max-model-len 98304 --limit_mm_per_prompt '{"image": 5}' --disable-cascade-attn --tensor-parallel-size 8 --gpu_memory_utilization 0.9 --pipeline-parallel-size 1 --host 0.0.0.0 --port 8000 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0}'
```

3. Run client:
```bash
# Same as 32B version
guidellm --data "prompt_tokens=512,generated_tokens=512,images=1,width=1770,height=1180" --data-type emulated --model /path/to/model --target "http://localhost:8000/v1" --max-requests 1
```

## Model Results

### Benchmarking vLLM

```bash
git clone https://github.com/vllm-project/vllm.git -b v0.8.3 --depth=1
python3 vllm/benchmarks/benchmark_throughput.py \
  --model {model_name} \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 10 \
  --hf-split train
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| Qwen2.5-VL | BF16 | 0.4 | 339.56 | 50.84 |