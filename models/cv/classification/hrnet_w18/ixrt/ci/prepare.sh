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

pip install -r ../../ixrt_common/requirements.txt
pip3 install --no-build-isolation mmcv==1.5.3 mmcls==0.24.0
unzip -q /root/data/repos/mmpretrain-0.24.0.zip -d ./
mkdir checkpoints
python3 ../../ixrt_common/export_mmcls.py --cfg mmpretrain/configs/hrnet/hrnet-w18_4xb32_in1k.py --weight hrnet-w18_3rdparty_8xb32_in1k_20220120-0c10b180.pth --output hrnet_w18.onnx

# Downgrade an ONNX model's IR version to 9 for onnxruntime <= 1.17.1
python3 ../../ixrt_common/make_ir9_model.py -i hrnet_w18.onnx -o hrnet_w18_ir9.onnx
mv hrnet_w18_ir9.onnx checkpoints/hrnet_w18.onnx