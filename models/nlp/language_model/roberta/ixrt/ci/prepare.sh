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

mkdir -p data
cp -r /root/data/checkpoints/open_roberta data/
cp /root/data/3rd_party/roberta-torch-fp32.json ./
cp /root/data/3rd_party/iluvatar-corex-ixrt/tools/optimizer/optimizer.py ./
# export onnx
python3 export_onnx.py --model_path open_roberta/roberta-base-squad.pt --output_path open_roberta/roberta-torch-fp32.onnx

# Simplify onnx model
onnxsim open_roberta/roberta-torch-fp32.onnx open_roberta/roberta.onnx

# Link and install requirements
ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# Move open_roberta
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
mv open_roberta ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/

# Get open_squad
cp /root/data/datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad

# Get csarron.tar
wget http://files.deepspark.org.cn:880/deepspark/csarron.tar
tar xf csarron.tar
rm -f csarron.tar
mv csarron/ ./ByteMLPerf/byte_infer_perf/