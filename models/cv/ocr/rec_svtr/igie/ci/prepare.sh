#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt

tar -xf rec_svtr_tiny_none_ctc_en_train.tar

git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git --depth=1

cd PaddleOCR

# Export the trained model into inference model
python3 tools/export_model.py -c ../rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=../rec_svtr_tiny_none_ctc_en_train/best_accuracy Global.save_inference_dir=./inference/rec_svtr_tiny

# Export the inference model to onnx model
paddle2onnx --model_dir ./inference/rec_svtr_tiny --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ../SVTR.onnx --opset_version 13 --enable_onnx_checker True

cd ..

# Use onnxsim optimize onnx model
onnxsim SVTR.onnx SVTR_opt.onnx

export DATASETS_DIR=./lmdb_evaluation/