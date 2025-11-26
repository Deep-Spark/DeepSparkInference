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

batchsize=${BATCH_SIZE:-"32"}
model_path="yolox"
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
DECODER_INPUT_NAMES="/head/obj_preds.0/Conv_output_0 /head/cls_preds.0/Conv_output_0 /head/reg_preds.1/Conv_output_0 /head/cls_preds.1/Conv_output_0 /head/reg_preds.2/Conv_output_0 /head/obj_preds.2/Conv_output_0 /head/cls_preds.2/Conv_output_0"

# cut onnx
python3 python/cut_model.py                             \
        --input_model ${model_path}.onnx                \
        --output_model ${model_path}_cut.onnx           \
        --input_names images                            \
        --output_names ${DECODER_INPUT_NAMES[@]}        

# create onnx
python3 python/deploy.py                        \
        --src ${model_path}_cut.onnx            \
        --dst ${model_path}_decoder.onnx

# build engine
python3 python/build_engine.py                  \
        --model ${model_path}.onnx              \
        --precision float16                     \
        --engine ${model_path}_decoder.engine

# inference
python3 python/inference.py                             \
        --engine ${model_path}_decoder.engine           \
        --batchsize ${batchsize}                        \
        --datasets ${datasets_path}

