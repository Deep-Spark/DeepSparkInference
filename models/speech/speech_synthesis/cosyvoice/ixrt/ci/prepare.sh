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
apt-get install sox libsox-dev
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 
pip install onnxsim
cp -r /root/data/repos/CosyVoice ./
cd CosyVoice
git checkout 1fc843514689daa61b471f1bc862893b3a5035a7
# cp modify files
cp -rf ../cosyvoice ./
cp ../asset/zero_shot_reference.wav ./asset/
cp -r ../scripts ./
cp ../build_dynamic_engine.py ./
cp ../inference.py ./

rm -rf pretrained_models
mkdir -p pretrained_models
ln -s /root/data/checkpoints/CosyVoice2-0.5B pretrained_models/
# remove old engine
rm ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp16.mygpu.plan
onnxsim ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32_sim.onnx