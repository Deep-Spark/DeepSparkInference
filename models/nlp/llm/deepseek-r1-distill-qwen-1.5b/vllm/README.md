# DeepSeek-R1-Distill-Qwen-1.5B (vLLM)

## Model Description

DeepSeek-R1-Distill models are fine-tuned based on open-source models, using samples generated by DeepSeek-R1. We
slightly change their configs and tokenizers.  We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints
based on Qwen2.5 and Llama3 series to the community.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0 | 25.09 |
| MR-V100 | 4.2.0 | 25.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>

```bash
cd deepseek-r1-distill-qwen-1.5b/vllm
mkdir -p data/
ln -s /path/to/DeepSeek-R1-Distill-Qwen-1.5B ./data/
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx
```

## Model Inference

### Inference with offline

```bash
python3 offline_inference.py --model ./data/DeepSeek-R1-Distill-Qwen-1.5B --max-tokens 256 -tp 1 --temperature 0.0 --max-model-len 3096
```

### Inference with serve

```bash
vllm serve data/DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --trust-remote-code
```

## Model Results

| Model                         | QPS    |
| :----: | :----: |
| DeepSeek-R1-Distill-Qwen-1.5B | 259.42 |

## References

- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
