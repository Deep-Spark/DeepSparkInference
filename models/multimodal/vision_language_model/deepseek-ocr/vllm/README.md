# DeepSeek-OCR (vLLM)

## Model Description

DeepSeek-OCR is DeepSeek's optical character recognition (OCR) system designed to extract text from images and documents. Here are its key features:

- Text Detection: Identifies text regions within images, PDFs, and other document formats
- Text Recognition: Converts detected text areas into machine-readable text
- Multi-format Support: Works with various file types including images (JPG, PNG, etc.) and PDF documents
- High Accuracy: Provides precise text extraction with advanced AI models

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>

```bash
cp -r ../../vllm_public_assets/ ./

# Download model from the website and make sure the model's path is "./deepseek-ocr"
```

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.


## Model Inference

```bash
python3 offline_inference_vision_language.py --model-type deepseek_ocr
```

## Model Results

### Benchmarking vLLM

```bash
vllm bench throughput --model ./deepseek-ocr --backend vllm-chat --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --num-prompts 10  --hf-split train --trust_remote_code
```

### Benchmarking Results

| Model | Precision  | QPS | Total TPS | Output TPS |
| :----: | :----: | :----: | :----: | :----: |
| DeepSeek-OCR | BF16 | 1.12 | 912.13 | 142.78 |