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

pip install -r requirements.txt

cp -r /root/data/3rd_party/lightweight-human-pose-estimation.pytorch ./
cd lightweight-human-pose-estimation.pytorch
mv scripts/convert_to_onnx.py .
ln -s /root/data/checkpoints/checkpoint_iter_370000.pth ./
python3 convert_to_onnx.py --checkpoint-path checkpoint_iter_370000.pth
cd ..
mkdir -p checkpoints
onnxsim ./lightweight-human-pose-estimation.pytorch/human-pose-estimation.onnx ./checkpoints/lightweight_openpose.onnx