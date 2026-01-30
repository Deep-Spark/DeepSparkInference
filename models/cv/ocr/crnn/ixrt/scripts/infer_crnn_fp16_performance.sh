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

# Run paraments
BSZ=1
TGT=-1
WARM_UP=5
LOOP_COUNT=10
RUN_MODE=FPS
PRECISION=float16

# Update arguments
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

PROJ_DIR=./
CHECKPOINTS_DIR="${PROJ_DIR}checkpoints"
RUN_DIR=./
ORIGINE_MODEL=${CHECKPOINTS_DIR}/crnn_sim_end.onnx
MODEL_NAME=crnn

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}

step=0
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim_end.onnx

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
BSZ_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${BSZ}.onnx
if [ -f $BSZ_MODEL ];then
    echo "  "Change Batchsize Skip, $BSZ_MODEL has been existed
else
    python3 ${RUN_DIR}/modify_batchsize.py --batch_size ${BSZ} \
        --origin_model ${SIM_MODEL} --output_model ${BSZ_MODEL}
    echo "  "Generate ${BSZ_MODEL}
fi



# Build Engine
let step++
echo;
echo [STEP ${step}] : Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --precision ${PRECISION}                \
        --model ${BSZ_MODEL}                    \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference.py     \
    --engine_file=${ENGINE_FILE}    \
    --warm_up=${WARM_UP}            \
    --loop_count ${LOOP_COUNT}      \
    --test_mode ${RUN_MODE}         \
    --fps_target ${TGT}             \
    --bsz ${BSZ}; check_status

exit ${EXIT_STATUS}