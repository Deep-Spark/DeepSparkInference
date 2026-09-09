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

python3 export.py --weight yolov3.pt --output yolov3.onnx
# Downgrade an ONNX model's IR version to 9 for onnxruntime <= 1.17.1
python3 ../../igie_common/make_ir9_model.py -i yolov3.onnx -o yolov3_ir9.onnx
# Use onnxsim optimize onnx model
onnxsim yolov3_ir9.onnx yolov3_opt.onnx
