# PaddleOCR-VL-1.5 (vLLM)

## Model Description

PaddleOCR-VL-1.5 is an advanced multi-task 0.9B VLM for robust in-the-wild document parsing. It achieves 94.5% accuracy on OmniDocBench v1.5 and introduces innovative approaches for irregular-shaped document localization.

Key features:
- **Ultra-Compact**: 0.9B parameters with high efficiency
- **Multi-Task**: Supports OCR, table, formula, chart, spotting, and seal recognition
- **Robust**: Handles scanning artifacts, skew, warping, screen photography, and illumination
- **Multilingual**: Extended support for Tibetan script and Bengali

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.06 |

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>
- Image: <https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png>

### Install Dependencies

```bash
pip install "transformers>=5.0.0"
pip install --upgrade mistral-common
```

## Model Inference

```bash
python3 offline_inference.py
```