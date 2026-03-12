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

mkdir -p checkpoints/
mv yolov9s.pt yolov9.pt

# set weights_only=False to be comaptible with pytorch 2.7
sed -i '781 s/map_location=\"cpu\")/map_location=\"cpu\", weights_only=False)/' /usr/local/lib/python3.10/site-packages/ultralytics/nn/tasks.py

python3 export.py --weight yolov9.pt --batch 32
onnxsim yolov9.onnx ./checkpoints/yolov9.onnx