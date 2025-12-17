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

pip3 install -r requirements.txt
pip3 install /mnt/deepspark/data/install/mmcv-2.1.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
# export onnx model
python3 export.py --weight retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth --cfg freeanchor_r50_fpn_1x_coco.py --output freeanchor_r50.onnx

# use onnxsim optimize onnx model
onnxsim freeanchor_r50.onnx freeanchor_r50_opt.onnx --overwrite-input-shape input:32,3,800,1344