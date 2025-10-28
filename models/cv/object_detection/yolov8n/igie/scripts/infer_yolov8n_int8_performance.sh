#!/bin/bash

# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

batchsize=32
model_path="yolov8n.onnx"
quantized_model_path="yolov8n_int8.onnx"
datasets_path=${DATASETS_DIR}

if [ ! -e $quantized_model_path ]; then
    # quantize model to int8
    python3 quantize.py                       \
        --model_path ${model_path}            \
        --out_path ${quantized_model_path}    \
        --batch ${batchsize}                  \
        --datasets ${datasets_path}
fi

# build engine
python3 ../../igie_common/build_engine.py   \
    --model_path ${quantized_model_path}     \
    --input images:${batchsize},3,640,640    \
    --precision int8                         \
    --engine_path yolov8n_bs_${batchsize}_int8.so

# inference
python3 inference.py                              \
    --engine yolov8n_bs_${batchsize}_int8.so       \
    --batchsize ${batchsize}                      \
    --input_name images                           \
    --datasets ${datasets_path}                   \
    --perf_only True
