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

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    apt install -y numactl
elif [[ ${ID} == "centos" ]]; then
    yum install -y numactl
else
    echo "Not Support Os"
fi

if [[ $(uname -m) == "aarch64" ]]; then
    echo "Architecture is aarch64."
    pip3 install --no-cache-dir --force-reinstall --upgrade --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn scikit-learn
    pip3 install numpy==1.26.4
    apt install -y libgomp1
fi

pip3 install -r requirements.txt
# reference: https://github.com/facebookresearch/fairseq/commit/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99
ln -s /mnt/deepspark/data/3rd_party/fairseq ../
# reference: https://github.com/omry/omegaconf/tree/v2.3.0
ln -s /mnt/deepspark/data/3rd_party/omegaconf ../
cp ../omegaconf.py ../omegaconf/
# reference: https://github.com/facebookresearch/hydra/tree/v1.3.2
ln -s /mnt/deepspark/data/3rd_party/hydra ../
cd ../
python3 setup.py build_ext --inplace
cd igie/
mkdir -p data/datasets/
mkdir -p data/checkpoints
cp -r /mnt/deepspark/data/datasets/corex-inference-data-4.0.0/checkpoints/transformer/wmt14.en-fr.joined-dict.transformer ./data/checkpoints/
ln -s /mnt/deepspark/data/datasets/corex-inference-data-4.0.0/datasets/wmt14.en-fr.joined-dict.newstest2014 ./data/datasets/