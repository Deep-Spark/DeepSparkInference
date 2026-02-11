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
pip3 install /mnt/deepspark/data/install/mmcv_full-1.7.0+corex.20250108131027-cp310-cp310-linux_x86_64.whl
pip3 install mmcls==0.24.0 mmengine
# set weights_only=False to be comaptible with pytorch 2.7 
sed -i '334 s/map_location)/map_location, weights_only=False)/' /usr/local/lib/python3.10/site-packages/mmcv/runner/checkpoint.py
unzip -q /mnt/deepspark/data/repos/mmpretrain-0.24.0.zip -d ./
python3 ../../igie_common/export_mmcls.py --cfg mmpretrain/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py --weight repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth --output repvgg.onnx