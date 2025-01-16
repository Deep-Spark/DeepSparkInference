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

pip install -r requirements.txt
unzip /root/data/repos/yolox-f00a798c8bf59f43ab557a2f3d566afa831c8887.zip -d ./
ln -s /root/data/checkpoints/yolox_m.pth ./YOLOX/
# install ixrt run
bash /root/data/3rd_party/ixrt-0.10.0+corex.4.2.0.20250115-linux_x86_64.run
cd YOLOX && python3 setup.py develop && python3 tools/export_onnx.py --output-name ../yolox.onnx -n yolox-m -c yolox_m.pth --batch-size 32
if [ "$1" = "nvidia" ]; then
    cd ../plugin && mkdir -p build && cd build && cmake .. -DUSE_TRT=1 && make -j12
else
    cd ../plugin && mkdir -p build && cd build && cmake .. -DIXRT_HOME=/usr/local/corex && make -j12
fi