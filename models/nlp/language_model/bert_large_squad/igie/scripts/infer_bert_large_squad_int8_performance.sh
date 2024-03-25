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
seqlen=256
input_shape=${batchsize},${seqlen}
model_path="bert_large_int8.hdf5"

# build engine
python3 build_engine.py                 \
    --model_path ${model_path}          \
    --input input_ids:${input_shape}    \
    --precision int8                    \
    --engine_path bert-large-squad


# inference
python3 inference.py                                     \
    --engine bert-large-squad                            \
    --batchsize ${batchsize}                             \
    --seqlen ${seqlen}                                   \
    --precision int8                                     \
    --datasets ${DATASETS_DIR}                           \
    --perf_only True