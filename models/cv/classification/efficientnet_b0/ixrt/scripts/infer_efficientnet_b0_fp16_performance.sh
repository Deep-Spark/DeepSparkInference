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
model_path="efficientnet_b0"
# model_path="resnet18"
datasets_path=${DATASETS_DIR}

# change batchsize
python3 modify_batchsize.py              \
        --batch_size ${batchsize}               \
        --origin_model ${model_path}.onnx     \
        --output_model ${model_path}_bs32.onnx

# build engine
python3 build_engine.py         \
        --model ${model_path}_bs32.onnx                     \
        --precision float16                    \
        --engine ${model_path}_fp16.engine

# inference
python3 inference.py                             \
        --test_mode FPS \
        --engine_file ${model_path}_fp16.engine       \
        --bsz ${batchsize}                        \
        --datasets_dir ${datasets_path}             \
        --warm_up 5                               \
        --loop_count 20                            