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

if [[ $(uname -m) == "aarch64" ]]; then
    echo "Architecture is aarch64."
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
fi

pip3 install -r requirements.txt

cp -r /mnt/deepspark/data/3rd_party/yolov7 ./
cd yolov7
# export onnx model
python3 export.py --weights ../yolov7.pt --simplify --img-size 640 640 --dynamic-batch --grid
