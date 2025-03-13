# MODEL_NAME (IGIE/IxRT/vLLM/TGI/TRT-LLM/IxFormer)

## Model Description

A brief introduction about this model.
A brief introduction about this model.
A brief introduction about this model.

## Supported Environments

| Iluvatar GPU | IXUCA SDK |
|--------------|-----------|
| MR-V100      | 4.2.0     |

## Model Preparation

### Prepare Resources

```bash
python3 dataset/coco/download_coco.py
```

Go to huggingface.

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install
```

### Model Conversion

## Model Inference

### FP16

```bash
bash test_fp16.sh
```

### INT8

```bash
bash test_int8.sh
```

## Model Results

| Model      | GPU        | Precision | Performance |
|------------|------------|-----------|-------------|
| MODEL_NAME | MR-V100 x1 |           |             |

## References

- [refer-page-name](https://refer-links)
- [Paper](Paper_link)
