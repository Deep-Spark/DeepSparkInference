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

pip install -r ../../ixrt_common/requirements.txt
pip install mmcv==1.5.3 mmcls==0.24.0 ppq pycuda transformers==4.37.1
unzip -q /root/data/repos/mmpretrain-0.24.0.zip -d ./
mkdir checkpoints
python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/resnet/resnetv1d50_b32x8_imagenet.py --weight resnetv1d50_b32x8_imagenet_20210531-db14775a.pth --output checkpoints/resnet_v1_d50.onnx