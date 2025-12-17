#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

EXIT_STATUS=0
check_status()
{
    ret_code=${PIPESTATUS[0]}
    if [ ${ret_code} != 0 ]; then
    [[ ${ret_code} -eq 10 && "${TEST_PERF:-1}" -eq 0 ]] || EXIT_STATUS=1
    fi
}

options=$@
arguments=($options)
index=0

BS=32
TGT=0.75

for argument in $options
do
    # Incrementing index
    index=`expr $index + 1`

    # The conditions
    case $argument in
      --bs) BS=${arguments[index]};;
      --tgt) TGT=${arguments[index]};;
    esac
done

python3 build_engine.py  \
    --model-path resnet50.onnx     \
    --data-path ${DATASETS_DIR}   \
    --model-format onnx          \
    --model-name resnet50           \
    --input-name "input"             \
    --model-layout NCHW             \
    --convert-layout NHWC           \
    --input-shape ${BS} 3 224 224   \
    --precision fp16

python3 inference.py  \
    --model-path resnet50.onnx     \
    --data-path ${DATASETS_DIR}   \
    --model-format onnx          \
    --model-name resnet50           \
    --input-name "input"             \
    --input-shape ${BS} 3 224 224   \
    --precision fp16                \
    --test-count 50000              \
    --workers 16                    \
    --acc1-target ${TGT};check_status
exit ${EXIT_STATUS}