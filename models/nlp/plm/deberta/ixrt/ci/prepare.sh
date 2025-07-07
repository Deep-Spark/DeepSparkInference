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

cp /root/data/3rd_party/deberta-torch-fp32.json ./
git clone https://gitee.com/deep-spark/iluvatar-corex-ixrt.git --depth=1
cp -r iluvatar-corex-ixrt/tools/optimizer/ ../../../../../toolbox/ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/

python3 torch2onnx.py --model_path /root/data/checkpoints/open_deberta/deberta-base-squad.pt --output_path deberta-torch-fp32.onnx
onnxsim deberta-torch-fp32.onnx deberta-torch-fp32-sim.onnx
python3 remove_clip_and_cast.py

mkdir -p data/open_deberta
cp ./deberta-sim-drop-clip-drop-invaild-cast.onnx data/open_deberta/deberta.onnx

ln -s ../../../../../toolbox/ByteMLPerf ./

pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# setup
cp /root/data/datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/

mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular
cp -r /root/data/checkpoints/open_deberta ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/
cp ./deberta-sim-drop-clip-drop-invaild-cast.onnx ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/popular/open_deberta/

cd ./ByteMLPerf/byte_infer_perf/general_perf
cp -r /root/data/3rd_party/workloads ./
# wget http://files.deepspark.org.cn:880/deepspark/Palak.tar
cp /root/data/3rd_party/Palak.tar ./
tar -zxvf Palak.tar

#接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py -AutoTokenizer.from_pretrained("Palak/microsoft_deberta-base_squad") => AutoTokenizer.from_pretrained("/Your/Path/Palak/microsoft_deberta-base_squad")

# run acc perf
sed -i 's/tensorrt_legacy/tensorrt/g' backends/ILUVATAR/common.py