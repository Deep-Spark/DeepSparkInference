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

pip3 install -r requirements.txt

cp /root/data/3rd_party/albert-torch-fp32.json ./
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

python3 torch2onnx.py --model_path /root/data/checkpoints/open_albert/albert-base-squad.pt --output_path albert-torch-fp32.onnx
onnxsim albert-torch-fp32.onnx albert-torch-fp32-sim.onnx

mkdir -p data/open_albert
mv ./albert-torch-fp32-sim.onnx data/open_albert/albert.onnx

# link and install requirements
ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# edit madlag/albert-base-v2-squad path
# sed -i "s#madlag#/${MODEL_PATH}/madlag#" ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py

# copy open_squad data
cp /root/data/datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/

# copy open_albert data
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_albert
cp /root/data/checkpoints/open_albert/*.pt ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_albert

# run acc script
cd ./ByteMLPerf/byte_infer_perf/general_perf
# wget http://files.deepspark.org.cn:880/deepspark/madlag.tar
cp /root/data/3rd_party/madlag.tar ./
tar xvf madlag.tar
rm -f madlag.tar
cp -r /root/data/3rd_party/workloads ./
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/common.py
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/compile_backend_iluvatar.py
sed -i 's/tensorrt_legacy/tensorrt/' ./backends/ILUVATAR/runtime_backend_iluvatar.py