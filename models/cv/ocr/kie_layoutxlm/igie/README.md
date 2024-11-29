# LayoutXLM

## Description

LayoutXLM is a groundbreaking multimodal pre-trained model for multilingual document understanding, achieving exceptional performance by integrating text, layout, and image data.

## Setup

```shell
pip3 install -r requirements.txt
```

## Download

Pretrained model: <https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar>

Dataset: <https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar> to download the XFUND_zh dataset.

## Model Conversion

```shell

tar -xf ser_vi_layoutxlm_xfund_pretrained.tar
tar -xf XFUND.tar

git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git --depth=1

cd PaddleOCR
mkdir -p train_data/XFUND
cp ../XFUND/class_list_xfun.txt train_data/XFUND

# Export the trained model into inference model
python3 tools/export_model.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=../ser_vi_layoutxlm_xfund_pretrained/best_accuracy Global.save_inference_dir=./inference/ser_vi_layoutxlm

# Export the inference model to onnx model
paddle2onnx --model_dir ./inference/ser_vi_layoutxlm --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ../kie_ser.onnx --opset_version 11 --enable_onnx_checker True

cd ..

# Use onnxsim optimize onnx model
onnxsim kie_ser.onnx kie_ser_opt.onnx
```

## Inference

```shell
export DATASETS_DIR=/Path/to/XFUND/
```

### FP16

```shell
# Accuracy
bash scripts/infer_kie_layoutxlm_fp16_accuracy.sh
# Performance
bash scripts/infer_kie_layoutxlm_fp16_performance.sh
```

## Results

| Model   | BatchSize | Precision | FPS    | Hmean  |
| ------- | --------- | --------- | ------ | ------ |
| Kie_ser | 8         | FP16      | 107.65 | 93.61% |

## Reference

PaddleOCR: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/kie/algorithm_kie_layoutxlm.md>
