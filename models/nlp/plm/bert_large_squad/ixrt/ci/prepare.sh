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

# install ixrt run
bash /root/data/install/ixrt-1.0.0.alpha+corex.4.3.0-linux_x86_64.run

if [ "$1" = "nvidia" ]; then
    cmake -S . -B build -DUSE_TENSORRT=true
    cmake --build build -j16
else
    cmake -S . -B build
    cmake --build build -j16
fi

pip install -r requirements.txt
mkdir -p ./python/data
ln -s /root/data/checkpoints/bert-large-uncased/ ./python/data && ln -s /root/data/datasets/squad/ ./python/data