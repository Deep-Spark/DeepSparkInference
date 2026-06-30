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

pip install -r ../../ixrt_common/requirements.txt
mkdir checkpoints
python3 export_onnx.py --origin_model /root/data/checkpoints/res2net50.pth --output_model res2net50.onnx

# Downgrade an ONNX model's IR version to 9 for onnxruntime <= 1.17.1
python3 ../../ixrt_common/make_ir9_model.py -i res2net50.onnx -o res2net50_ir9.onnx
mv res2net50_ir9.onnx checkpoints/res2net50.onnx