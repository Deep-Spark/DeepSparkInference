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

cp /root/data/3rd_party/deberta-torch-fp32.json ./
cp /root/data/3rd_party/iluvatar-corex-ixrt/tools/optimizer/optimizer.py ./
python3 torch2onnx.py --model_path /root/data/checkpoints/open_deberta/deberta-base-squad.pt --output_path deberta-torch-fp32.onnx
onnxsim deberta-torch-fp32.onnx deberta-torch-fp32-sim.onnx
python3 remove_clip_and_cast.py

mkdir -p data/open_deberta
mv ./deberta-sim-drop-clip-drop-invaild-cast.onnx data/open_deberta/deberta.onnx

ln -s ../../../../../toolbox/ByteMLPerf ./

pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/backends/ILUVATAR/requirements.txt

# setup
cp /root/data/datasets/open_squad/* ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/

cp ./deberta-sim-drop-clip-drop-invaild-cast.onnx /root/data/checkpoints/open_deberta/
cp -r /root/data/checkpoints/open_deberta ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/

cd ./ByteMLPerf/byte_infer_perf/general_perf
wget http://files.deepspark.org.cn:880/deepspark/Palak.tar
tar -zxvf Palak.tar

#接着修改代码：ByteMLPerf/byte_infer_perf/general_perf/datasets/open_squad/data_loader.py -AutoTokenizer.from_pretrained("Palak/microsoft_deberta-base_squad") => AutoTokenizer.from_pretrained("/Your/Path/Palak/microsoft_deberta-base_squad")

# run acc perf
sed -i 's/tensorrt_legacy/tensorrt/g' backends/ILUVATAR/common.py