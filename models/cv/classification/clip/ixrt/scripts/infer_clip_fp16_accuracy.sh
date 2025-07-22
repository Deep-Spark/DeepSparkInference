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
BSZ=32
TGT=-1
WARM_UP=0
LOOP_COUNT=-1
RUN_MODE=ACC
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

source ${CONFIG_DIR}
ORIGINE_MODEL=${CHECKPOINTS_DIR}/${ORIGINE_MODEL}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo CONFIG_DIR : ${CONFIG_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}

step=0

# Change Input size and Simplify Model
let step++
echo;
echo [STEP ${step}] : Change Batchsize and Simplify
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${BSZ}_sim.onnx
if [ -f $SIM_MODEL ];then
    echo "  "Change Input size and Simplify Model Skip, $SIM_MODEL has been existed
else
    onnxsim ${ORIGINE_MODEL} ${SIM_MODEL} \
        --overwrite-input-shape input_ids:1000,22 pixel_values:${BSZ},3,224,224 attention_mask:1000,22
    echo "  "Generate ${SIM_MODEL}
fi

# optimizer
let step++
echo;
echo [STEP ${step}] : Optimizer
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${BSZ}_sim_end.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Optimizer Model Skip, $FINAL_MODEL has been existed
else
    python3 ${OPTIMIER_FILE} --onnx ${SIM_MODEL} --model_type vit
    echo "  "Generate ${FINAL_MODEL}
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
        --model ${FINAL_MODEL}                    \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference.py     \
    --engine_file=${ENGINE_FILE}    \
    --datasets_dir=${DATASETS_DIR}  \
    --imgsz=${IMGSIZE}              \
    --warm_up=${WARM_UP}            \
    --loop_count ${LOOP_COUNT}      \
    --test_mode ${RUN_MODE}         \
    --acc_target ${TGT}             \
    --bsz ${BSZ}; check_status

exit ${EXIT_STATUS}
