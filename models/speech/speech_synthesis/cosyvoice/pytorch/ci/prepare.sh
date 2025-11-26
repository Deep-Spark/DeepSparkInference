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
apt update
apt-get install sox libsox-dev
pip3 install -r requirements.txt
pip3 install onnxruntime==1.18.0
cp -r /mnt/deepspark/data/repos/CosyVoice ./
cd CosyVoice
mkdir -p pretrained_models
ln -s /mnt/deepspark/data/checkpoints/CosyVoice2-0.5B pretrained_models/

cp ../inference_test.py ./