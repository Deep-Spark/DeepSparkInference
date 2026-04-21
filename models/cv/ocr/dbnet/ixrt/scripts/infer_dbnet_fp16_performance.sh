#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

BSZ=16
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

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
	EXIT_STATUS=1
    fi
}

DATASETS_DIR="/root/data/datasets/icdar_2015/icdar_2015_images"
CHECKPOINTS_DIR="./checkpoints"
RUN_DIR="${RUN_DIR:-.}"

python3 ${RUN_DIR}/build_engine.py --model=${CHECKPOINTS_DIR}/r50_en_dbnet.onnx\
                        --engine=${CHECKPOINTS_DIR}/float16_r50_en_dbnet.engine\
                        --batch_size=${BSZ}\
                        --precision="float16"

python3 ${RUN_DIR}/dbnet_inference.py \
        --engine_file ${CHECKPOINTS_DIR}/float16_r50_en_dbnet.engine \
        --target "perf" \
        --batch_size ${BSZ} \
        --target_fps ${TGT};check_status
        
 exit ${EXIT_STATUS}