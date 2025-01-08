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

# clone yolov4
git clone --depth 1 https://github.com/Tianxiaomo/pytorch-YOLOv4.git yolov4

mkdir data
# export onnx model
python3 export.py --cfg yolov4/cfg/yolov4.cfg --weight /root/data/checkpoints/yolov4.weights --batchsize 16 --output data/yolov4.onnx
mv yolov4_16_3_608_608_static.onnx data/yolov4.onnx

# Use onnxsim optimize onnx model
onnxsim data/yolov4.onnx data/yolov4_sim.onnx
