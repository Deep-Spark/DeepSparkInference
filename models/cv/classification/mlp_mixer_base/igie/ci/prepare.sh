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
unzip -q /mnt/deepspark/data/repos/mmpretrain-0.24.0.zip -d ./
# export onnx model
python3 export.py --cfg mmpretrain/configs/mlp_mixer/mlp-mixer-base-p16_64xb64_in1k.py --weight mixer-base-p16_3rdparty_64xb64_in1k_20211124-1377e3e0.pth --output mlp_mixer_base.onnx

# Use onnxsim optimize onnx model
onnxsim mlp_mixer_base.onnx mlp_mixer_base_opt.onnx