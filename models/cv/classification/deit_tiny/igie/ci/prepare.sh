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

apt install -y libgl1-mesa-glx
pip3 install -r requirements.txt
unzip /mnt/deepspark/volumes/mdb/data/repos/mmpretrain-0.24.0.zip -d ./
python3 export.py --cfg mmpretrain/configs/deit/deit-tiny_pt-4xb256_in1k.py --weight deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth --output deit_tiny.onnx
onnxsim deit_tiny.onnx deit_tiny_opt.onnx