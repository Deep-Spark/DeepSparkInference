# DeepSeek-OCR (Transformers)

## Model Description

DeepSeek-OCR is DeepSeek's optical character recognition (OCR) system designed to extract text from images and documents. Here are its key features:

- Text Detection: Identifies text regions within images, PDFs, and other document formats
- Text Recognition: Converts detected text areas into machine-readable text
- Multi-format Support: Works with various file types including images (JPG, PNG, etc.) and PDF documents
- High Accuracy: Provides precise text extraction with advanced AI models

## Supported Environments

| GPU | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release | Branch |
| :----: | :----: | :----: | :----: |
| MR-V100 | 4.4.0 | 26.03 | release/26.03 |
| MR-V100 | 4.3.0 | 25.12 | release/25.12 |

> **Note:** 请切换到与您的 SDK 版本对应的 Release 分支进行测试。请勿直接在 master 分支上运行测试，因为 master 分支可能包含与您的本地 SDK 版本不兼容的最新更改。
>
> 切换分支命令示例：`git checkout release/26.03`

## Model Preparation

### Prepare Resources

- Model: <https://huggingface.co/deepseek-ai/DeepSeek-OCR>

### Install Dependencies

In order to run the model smoothly, you need to get the sdk from [resource center](https://support.iluvatar.com/#/ProductLine?id=2) of Iluvatar CoreX official website.

```bash
pip install transformers==4.46.3 einops easydict addict matplotlib
```

## Model Inference

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/
# Change image_file and output_path value with your own config
python3 run_dpsk_ocr.py
```

## Model Results