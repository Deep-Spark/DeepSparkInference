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

apt install -y libnuma-dev

pip install -r requirements.txt
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

mkdir -p data
cp -r /root/data/checkpoints/open_videobert data/

# link and install requirements
ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# copy data
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cifar/
cp -r /root/data/datasets/open_cifar/cifar-100-python/ ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cifar/
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_videobert/
cp /root/data/checkpoints/open_videobert/videobert.onnx ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_videobert/video-bert.onnx
cd ./ByteMLPerf/byte_infer_perf/general_perf
cp -r /root/data/3rd_party/workloads ./
