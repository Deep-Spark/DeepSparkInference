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
cp -r /root/data/checkpoints/open_wide_deep_saved_model ./
python3 export_onnx.py --model_path open_wide_deep_saved_model --output_path open_wide_deep_saved_model/widedeep.onnx

# Simplify onnx model
onnxsim open_wide_deep_saved_model/widedeep.onnx open_wide_deep_saved_model/widedeep_sim.onnx
python3 deploy.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx
python3 change2dynamic.py --model_path open_wide_deep_saved_model/widedeep_sim.onnx --output_path open_wide_deep_saved_model/widedeep_sim.onnx

mkdir -p data/open_widedeep
mv open_wide_deep_saved_model/widedeep_sim.onnx data/open_widedeep/widedeep.onnx

# link and install ByteMLPerf requirements
ln -s ../../../../../toolbox/ByteMLPerf ./
pip3 install -r ./ByteMLPerf/byte_infer_perf/general_perf/requirements.txt

# Get eval.csv and onnx
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model
mkdir -p ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/

cp /root/data/datasets/eval.csv ./ByteMLPerf/byte_infer_perf/general_perf/datasets/open_criteo_kaggle/

wget http://files.deepspark.org.cn:880/deepspark/widedeep_dynamicshape_new.onnx
cp open_wide_deep_saved_model/* ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/
mv widedeep_dynamicshape_new.onnx ./ByteMLPerf/byte_infer_perf/general_perf/model_zoo/regular/open_wide_deep_saved_model/widedeep_dynamicshape.onnx

cp -r /root/data/3rd_party/workloads ./ByteMLPerf/byte_infer_perf/general_perf/