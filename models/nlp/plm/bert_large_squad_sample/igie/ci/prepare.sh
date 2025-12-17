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

mkdir -p ./data/checkpoints/bert_large_squad
mkdir -p ./data/datasets/bert_large_squad
ln -s /mnt/deepspark/data/checkpoints/bert-large-uncased ./data/checkpoints/bert_large_squad/
ln -s /mnt/deepspark/data/datasets/squad ./data/datasets/bert_large_squad/

if [ -f /etc/redhat-release ]; then
    if grep -qi "CentOS" /etc/redhat-release; then
        yum install -y numactl
    fi
elif [ -f /etc/system-release ]; then
    if grep -qi "Kylin" /etc/system-release; then
        yum install -y numactl
    fi
else
    apt install -y numactl
fi

pip3 install --no-dependencies transformers
pip3 install datasets onnx tabulate pycuda