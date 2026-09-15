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
# set weights_only=False to be comaptible with pytorch 2.7
sed -i '716 s/\"cpu\")/\"cpu\", weights_only=False)/' /usr/local/lib/python3.12/site-packages/ultralytics/nn/tasks.py
python3 export.py --weight yolov8n-face.pt --batch 32

git clone https://github.com/wondervictor/WiderFace-Evaluation.git widerface_evaluate

# official eval_tools (contains ground_truth/*.mat)
wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
unzip eval_tools.zip
mkdir -p ground_truth
cp eval_tools/ground_truth/wider_*.mat ground_truth/