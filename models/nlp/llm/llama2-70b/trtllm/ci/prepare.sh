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
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y libgl1-mesa-glx
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL
else
    echo "Not Support Os"
fi

bash scripts/set_environment.sh .

# Download model from the website and make sure the model's path is "data/llama2-70b-chat"
# Download dataset from the website and make sure the dataset's path is "data/datasets_cnn_dailymail"
mkdir -p data
ln -s /mnt/deepspark/data/checkpoints/llama2-70b-chat data/llama2-70b-chat
ln -s /mnt/deepspark/data/datasets/datasets_cnn_dailymail data/datasets_cnn_dailymail
# Please download rouge.py to this path if your server can't attach huggingface.co.
mkdir -p rouge/
cp /mnt/deepspark/data/3rd_party/rouge.py rouge/