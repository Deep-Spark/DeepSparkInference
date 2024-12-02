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

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y libgl1-mesa-glx
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL
else
    echo "Not Support Os"
fi

pip3 install -r requirements.txt

tar -xf ser_vi_layoutxlm_xfund_pretrained.tar
tar -xf XFUND.tar

# clone release/2.6 PaddleOCR first
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