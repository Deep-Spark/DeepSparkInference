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

mkdir -p data
cp -r /root/data/checkpoints/open_roformer data/

# export onnx
python3 export_onnx.py --model_path ./data/open_roformer --output_path ./data/open_roformer/roformer-frozen_org.onnx

# Simplify onnx model
onnxsim ./data/open_roformer/roformer-frozen_org.onnx ./data/open_roformer/roformer-frozen.onnx
python3 deploy.py --model_path ./data/open_roformer/roformer-frozen.onnx --output_path ./data/open_roformer/roformer.onnx

# link ByteMLPerf and install requirements
ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt

# Comment Line102 in compile_backend_iluvatar.py
sed -i '102s/build_engine/# build_engine/' ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/compile_backend_iluvatar.py

# Move open_roformer
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
mv ./data/open_roformer ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/

# Setup open_cail2019 dataset
cp /root/data/datasets/open_cail2019/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_cail2019

# Go to general_perf/
cd ./ByteMLPerf/byte_infer_perf/general_perf
cp -r /root/data/3rd_party/workloads ./
# Modify model_zoo/roformer-tf-fp32.json
sed -i 's/segment:0/segment0/g; s/token:0/token0/g' model_zoo/roformer-tf-fp32.json