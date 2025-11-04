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

pip3 install -r requirements.txt

# install fast-reid
git clone https://github.com/JDAI-CV/fast-reid.git --depth=1
cd fast-reid
pip3 install -r docs/requirements.txt

# export onnx model
python3 tools/deploy/onnx_export.py --batch-size 32 --config-file configs/VehicleID/bagtricks_R50-ibn.yml --name fast_reid --output ../ --opts MODEL.WEIGHTS ../vehicleid_bot_R50-ibn.pth

cd ..