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

apt install -y libnuma-dev
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

pip install -r requirements.txt
mkdir -p general_perf/model_zoo/regular
mkdir -p general_perf/model_zoo/popular
mkdir -p general_perf/model_zoo/sota

cp /root/data/3rd_party/swin-large-torch-fp32.json ./
cp -r /root/data/checkpoints/swin-large ./general_perf/model_zoo/popular/

python3 torch2onnx.py --model_path ./general_perf/model_zoo/popular/swin-large/swin-transformer-large.pt --output_path swin-large-torch-fp32.onnx

ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# copy data
cp -r /root/data/datasets/open_imagenet/* ByteMLPerf/byte_infer_perf/general_perf/datasets/open_imagenet/
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/swin-large
cp general_perf/model_zoo/popular/swin-large/* ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/swin-large

cp -r /root/data/3rd_party/workloads ./ByteMLPerf/byte_infer_perf/general_perf/