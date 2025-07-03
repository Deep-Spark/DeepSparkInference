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

pip install /root/data/install/mmcv_full-1.7.0+corex.20250108131027-cp310-cp310-linux_x86_64.whl

mkdir -p checkpoints
ln -s /root/data/checkpoints/solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth ./
ln -s /root/data/datasets/coco2017 ./
python3 solo_torch2onnx.py --cfg ./solo_r50_fpn_3x_coco.py --checkpoint ./solo_r50_fpn_3x_coco_20210901_012353-11d224d7.pth --batch_size 1
mv r50_solo_bs1_800x800.onnx ./checkpoints/r50_solo_bs1_800x800.onnx