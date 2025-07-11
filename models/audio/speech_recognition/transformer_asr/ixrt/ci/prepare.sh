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
    apt install sox libsox-fmt-all
elif [[ ${ID} == "centos" ]]; then
    yum install -y mesa-libGL
    yum install sox sox-devel -y
else
    echo "Not Support Os"
fi

pip3 install -r requirements.txt

mkdir -p results/transformer
cp -r /root/data/checkpoints/8886 results/transformer/
mkdir -p results/transformer/8886/save
mkdir -p /home/data/speechbrain/aishell/csv_data
ln -s /root/data/datasets/AISHELL/data_aishell /home/data/speechbrain/aishell/
cp /root/data/datasets/rirs_noises.zip /home/data/speechbrain/aishell/
unzip -q /home/data/speechbrain/aishell/rirs_noises.zip -d /home/data/speechbrain/aishell/
cp results/transformer/8886/*.csv /home/data/speechbrain/aishell/csv_data

bash build.sh

python3 builder.py \
--ckpt_path results/transformer/8886/save \
--head_num 4 \
--max_batch_size 64  \
--max_seq_len 1024 \
--engine_path transformer.engine