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

mkdir -p checkpoints
unzip -q /root/data/repos/mmpretrain-0.24.0.zip -d ./
cd mmdetection
python3 tools/deployment/pytorch2onnx.py \
    ../fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py \
    /root/data/checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth \
    --output-file ../checkpoints/r50_fcos.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 800 800 \
    --show \
    --verify \
    --skip-postprocess \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1