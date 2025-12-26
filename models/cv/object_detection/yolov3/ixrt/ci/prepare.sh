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

pip3 install -r ../../ixrt_common/requirements.txt
mkdir checkpoints
unzip -q /root/data/3rd_party/onnx_tflite_yolov3.zip -d ./
cp /root/data/checkpoints/yolov3.weights onnx_tflite_yolov3/weights
cd onnx_tflite_yolov3
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
mv weights/export.onnx ../checkpoints/yolov3.onnx
cd ..
cp config/YOLOV3_CONFIG ../../ixrt_common/config/YOLOV3_CONFIG