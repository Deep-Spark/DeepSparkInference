#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

pip3 install -r requirements.txt
pip3 install open_clip_torch
pip3 install timm

python3 export.py --model-name ViT-B-32 --weight ViT-B-32.pt --output vit_b_32.onnx

onnxsim vit_b_32.onnx vit_b_32_opt.onnx
