# LLaVA-NeXT-based (vLLM)

## Model Description

E5-V is fine-tuned based on lmms-lab/llama3-llava-next-8b.

We propose a framework, called E5-V, to adpat MLLMs for achieving multimodal embeddings. E5-V effectively bridges the modality gap between different types of inputs, demonstrating strong performance in multimodal embeddings even without fine-tuning. We also propose a single modality training approach for E5-V, where the model is trained exclusively on text pairs, demonstrating better performance than multimodal training.

More details can be found in https://github.com/kongds/E5-V

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/royokong/e5-v>

```bash
cp -r ../../vllm_public_assets/ ./
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

## Model Inference

```bash
python3 offline_inference_vision_language_embedding.py  --model /path/to/e5-v/  --modality "image" --tensor_parallel_size 1 --task "embed" --trust_remote_code --max_model_len 4096
```

## Model Results