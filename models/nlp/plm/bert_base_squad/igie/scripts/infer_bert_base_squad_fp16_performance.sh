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

batchsize=8
seqlen=256
input_shape=${batchsize},${seqlen}
model_path="bert-base-uncased-squad-v1.onnx"

# build engine
python3 build_engine.py                                                                             \
    --model_path ${model_path}                                                                      \
    --input input_ids:${input_shape} attention_mask:${input_shape} token_type_ids:${input_shape}    \
    --precision fp16                                                                                \
    --engine_path bert-base-uncased-squad-v1_bs_${batchsize}_fp16.so


# inference
python3 inference.py                                                 \
    --engine bert-base-uncased-squad-v1_bs_${batchsize}_fp16.so      \
    --batchsize ${batchsize}                                         \
    --seqlen ${seqlen}                                               \
    --input_name input_ids attention_mask token_type_ids             \
    --datasets ${DATASETS_DIR}                                      \
    --perf_only True