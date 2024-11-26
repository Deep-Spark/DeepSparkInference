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

pip install -r requirements.txt
mkdir -p checkpoints
unzip -q /root/data/repos/mmpretrain-0.24.0.zip -d ./checkpoints/
python3 export_onnx.py   \
--config_file ./checkpoints/mmpretrain/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py  \
--checkpoint_file  /root/data/checkpoints/shufflenet_v1.pth \
--output_model ./checkpoints/shufflenet_v1.onnx