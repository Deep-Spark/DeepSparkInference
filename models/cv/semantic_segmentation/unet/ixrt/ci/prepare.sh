#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

pip3 install onnxsim
pip3 install opencv-python==4.6.0.66
pip3 uninstall mmcv-full mmcv -y
pip3 install mmcv==1.5.3
pip3 install prettytable
pip3 install onnx

mkdir -p checkpoints
ln -s /root/data/checkpoints/unet_export.onnx checkpoints/
# deploy, will generate unet.onnx
python3 deploy.py --onnx_name checkpoints/unet_export.onnx --save_dir checkpoints/ --data_type float16 "$@"

mkdir -p datasets
ln -s /root/data/datasets/DRIVE datasets/