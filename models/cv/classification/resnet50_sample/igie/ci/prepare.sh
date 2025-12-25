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

if [ -f /etc/redhat-release ]; then
    if grep -qi "CentOS" /etc/redhat-release; then
        yum install -y numactl
    fi
elif [ -f /etc/system-release ]; then
    if grep -qi "Kylin" /etc/system-release; then
        yum install -y numactl
    fi
else
    apt install numactl
fi

pip3 install pycocotools pytest opencv-python==4.6.0.66 tqdm
ln -s /mnt/deepspark/data/checkpoints/resnet50.onnx ./
ln -s /mnt/deepspark/data/checkpoints/resnet50-fp32.pt ./