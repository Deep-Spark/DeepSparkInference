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
# before install mmpose==1.3.1 need to install chchumpy==0.70 which is too older that is not compatible with newer Python versions or pip
# so need to downgrade pip to version 20.2.4
pip install pip==20.2.4
pip install mmpose==1.3.1
pip install --upgrade pip
pip install -r requirements.txt

# export onnx model
python3 export.py --weight rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth --cfg rtmpose-m_8xb256-420e_coco-256x192.py --output rtmpose.onnx

# use onnxsim optimize onnx model
onnxsim rtmpose.onnx rtmpose_opt.onnx