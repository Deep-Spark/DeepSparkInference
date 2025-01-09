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

mkdir checkpoints
cp -r /root/data/3rd_party/yolov5 ./
# 切换到需要的版本分支
cd yolov5

# 有一些环境需要安装
# wget https://ultralytics.com/assets/Arial.ttf
mkdir -p /root/.config/Ultralytics
cp /root/data/3rd_party/Arial.ttf /root/.config/Ultralytics/Arial.ttf

# 转换为onnx (具体实现可以参考 export.py 中的 export_onnx 函数)
python3 export.py --weights /root/data/checkpoints/yolov5m.pt --include onnx --opset 11 --batch-size 32
mv yolov5m.onnx ../checkpoints
cd ..
