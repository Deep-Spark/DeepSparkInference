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
#    under the License.i

batchsize=32
model_path="SVTR_opt.onnx"
datasets_path=${DATASETS_DIR}

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) batchsize=${arguments[index]};;
    esac
done

echo "batch size is ${batchsize}"

# build engine
python3 build_engine.py                            \
    --model_path ${model_path}                     \
    --input x:${batchsize},3,64,256                \
    --precision fp16                               \
    --engine_path svtr_bs_${batchsize}_fp16.so


# inference
python3 inference.py                              \
    --engine svtr_bs_${batchsize}_fp16.so         \
    --batchsize ${batchsize}                      \
    --input_name x                                \
    --datasets ${datasets_path}                   \
    --perf_only True