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
model_path="yolox"
datasets_path=${DATASETS_DIR}

# quant
python3 python/quant.py             \
        --model_name ${model_path}  \
        --model ${model_path}.onnx      \
        --dataset_dir ${datasets_path}/val2017      \
        --ann_file ${datasets_path}/annotations/instances_val2017.json      \
        --save_dir ./

# build engine
python3 python/build_engine_by_write_qparams.py         \
        --onnx quantized_yolox.onnx                     \
        --qparam_json quant_cfg.json                    \
        --engine ${model_path}_int8.engine

# inference
python3 python/inference.py                             \
        --engine ${model_path}_int8.engine              \
        --batchsize ${batchsize}                        \
        --datasets ${datasets_path}                     \
        --perf_only True                                \
        --loop_count 20