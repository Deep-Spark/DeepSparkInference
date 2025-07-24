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
pip3 install -r ../../igie_common/requirements.txt
pip3 install mmcv==1.5.3 mmcls==0.24.0
pip3 install http://sw.iluvatar.ai/download/corex/daily_packages/ivcore11/x86_64/20250723/apps/py3.10/igie-0.18.1+corex.4.3.0.20250723-cp310-cp310-linux_x86_64.whl
unzip -q /mnt/deepspark/data/repos/mmpretrain-0.24.0.zip -d ./
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/mvit/mvitv2-base_8xb256_in1k.py --weight mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth --output mvitv2_base.onnx
onnxsim mvitv2_base.onnx mvitv2_base_opt.onnx