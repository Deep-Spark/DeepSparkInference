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
unzip -q /root/data/repos/mmpretrain-0.24.0.zip -d ./
mkdir -p checkpoints
python3 export_onnx.py   \
    --config_file ./mmpretrain/configs/cspnet/cspresnet50_8xb32_in1k.py  \
    --checkpoint_file  /root/data/checkpoints/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth \
    --output_model ./checkpoints/cspresnet50.onnx
