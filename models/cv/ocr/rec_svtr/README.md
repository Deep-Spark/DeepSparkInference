# SVTR
## Description
SVTR proposes a single vision model for scene text recognition. This model completely abandons sequence modeling within the patch-wise image tokenization framework. Under the premise of competitive accuracy, the model has fewer parameters and faster speed.

## Setup
```shell
pip3 install tqdm
pip3 install onnx
pip3 install onnxsim
pip3 install numpy==1.24.0
pip3 install Polygon3
pip3 install paddlenlp==2.8.1
pip3 install lanms-neo==1.0.2
pip3 install paddleocr==2.8.1
pip3 install paddle2onnx
pip3 install python-bidi
```

## Download
Pretrained model:<https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar>

Dataset: <https://www.dropbox.com/scl/fo/zf04eicju8vbo4s6wobpq/ALAXXq2iwR6wKJyaybRmHiI?rlkey=2rywtkyuz67b20hk58zkfhh2r&e=1&dl=0> to download the lmdb evaluation datasets.

## Model Conversion
```shell
tar -xf rec_svtr_tiny_none_ctc_en_train.tar

git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git

cd PaddleOCR

# Export the trained model into inference model
python3 tools/export_model.py -c ../rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=../rec_svtr_tiny_none_ctc_en_train/best_accuracy Global.save_inference_dir=./inference/rec_svtr_tiny

# Export the inference model to onnx model
paddle2onnx --model_dir ./inference/rec_svtr_tiny --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ../SVTR.onnx --opset_version 13 --enable_onnx_checker True

cd ..

# Use onnxsim optimize onnx model
onnxsim SVTR.onnx SVTR_opt.onnx
``` 

## Inference
```shell 
export DATASETS_DIR=/Path/to/lmdb_evaluation/
```
### FP16
```shell
# Accuracy
bash scripts/infer_svtr_fp16_accuracy.sh
# Performance
bash scripts/infer_svtr_fp16_performance.sh
```

## Results
Model   |BatchSize  |Precision |FPS       |Acc       |
--------|-----------|----------|----------|----------|
SVTR    |    32     |   FP16   | 4936.47  |  88.29%  |

## Reference
PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/algorithm/text_recognition/algorithm_rec_svtr.md