#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 
pip3 install onnxsim==0.4.36
cp -r /mnt/deepspark/data/repos/CosyVoice ./
cd CosyVoice
git checkout ace7c47
# cp modify files
cp -r ../cosyvoice/cosyvoice ./
cp -r ../cosyvoice/examples ./
cp ../cosyvoice/example.py ./

rm -rf pretrained_models
mkdir -p pretrained_models
ln -s /mnt/deepspark/data/checkpoints/CosyVoice2-0.5B pretrained_models/
onnxsim ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.onnx ./pretrained_models/CosyVoice2-0.5B/flow.decoder.estimator.fp32.opt.onnx