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
unzip -q /root/data/checkpoints/20180408-102900.zip -d ./
unzip -q /root/data/datasets/facenet_datasets.zip -d ./
mkdir -p checkpoints
mkdir -p facenet_weights
cp -r /root/data/3rd_party/facenet-pytorch ./
cp ./tensorflow2pytorch.py facenet-pytorch
python3 ./facenet-pytorch/tensorflow2pytorch.py \
        --facenet_weights_path ./facenet_weights \
        --facenet_pb_path ./20180408-102900 \
        --onnx_save_name facenet_export.onnx
mv facenet_export.onnx ./facenet_weights

sed -i -e 's#/last_bn/BatchNormalization_output_0#1187#g' -e 's#/avgpool_1a/GlobalAveragePool_output_0#1178#g' deploy.py build_engine.py