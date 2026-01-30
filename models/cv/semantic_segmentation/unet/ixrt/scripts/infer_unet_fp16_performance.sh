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
    if ((${PIPESTATUS[0]} != 0));then
        EXIT_STATUS=1
    fi
}
TGT=-1
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --tgt) TGT=${arguments[index]};;
    esac
done

CHECKPOINTS_DIR=./checkpoints

# build engine
python3 ./build_engine.py          \
    --precision "float16"               \
    --model ${CHECKPOINTS_DIR}/unet.onnx                    \
    --engine ${CHECKPOINTS_DIR}/unet.engine

# infer
python3 -u main.py \
    --data_type float16 \
    --model_type unet \
    --engine_file ${CHECKPOINTS_DIR}/unet.engine \
    --bsz 1 \
    --imgh 64 \
    --fps_target ${TGT}\
    --imgw 64 \
    --test_mode FPS \
    --warm_up 10 \
    --run_loop 20 \
    --device 0  "$@";check_status