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

if [ -f /etc/system-release ]; then
    if grep -qi "Kylin" /etc/system-release; then
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
    fi
fi

pip3 install -r ../../ixrt_common/requirements.txt
mkdir -p checkpoints
cp -r /root/data/3rd_party/yolov7 ./
cd yolov7
ln -s /root/data/checkpoints/yolov7.pt ./
python3 export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 --batch-size 16
mv yolov7.onnx ../checkpoints/yolov7m.onnx
cd ..
