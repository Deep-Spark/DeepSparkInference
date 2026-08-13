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

pip3 install -r requirements.txt
pip3 install numpy==1.26.4

python3 export.py --weight yolov5m.pt --output yolov5m.onnx
# Downgrade an ONNX model's IR version to 9 for onnxruntime <= 1.17.1
python3 make_ir9_model.py -i yolov5m.onnx -o yolov5m_ir9.onnx

# Use onnxsim optimize onnx model
onnxsim yolov5m_ir9.onnx yolov5m_opt.onnx
