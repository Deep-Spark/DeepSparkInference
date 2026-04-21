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

set -e

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

MODEL_NAME="ddrnet"
BSZ=4
PRECISION="int8"
DEVICE=0
FORCE_BUILD=0
TGT_0=-1
TGT_1=-1

index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      -p | --precision) PRECISION=${arguments[index]};;
      -d | --device) DEVICE=${arguments[index]};;
      --bs) BSZ=${arguments[index]};;
      --tgt_iou) TGT_0=${arguments[index]};;
      --tgt_acc) TGT_1=${arguments[index]};;
      -f | --force) FORCE_BUILD=1;;
    esac
done

CHECKPOINTS_DIR="./checkpoints"
DATASET_DIR="/root/data/datasets"
LIST_PATH="/root/data/datasets/cityscapes/val.lst"
RUN_DIR="${RUN_DIR:-.}"
ORIGINE_MODEL="${CHECKPOINTS_DIR}/ddrnet23.onnx"

echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}
echo;

function run_cmd()
{
    echo "[CMD]: $@"
    eval $@
}

step=1

# Simplify Model
echo [STEP ${step}] : Simplify Model
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model, ${SIM_MODEL} has been existed
else
    run_cmd python3 ${RUN_DIR}/sim_onnx_model.py    \
                --raw_model_path ${ORIGINE_MODEL}   \
                --sim_model_path ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi
let step++
echo;

# Quant Model
if [ $PRECISION == "int8" ];then
    echo [STEP ${step}] : Quant Model
    QUANT_MODEL=${CHECKPOINTS_DIR}/quantized_${MODEL_NAME}_sim.onnx
    if [ -f ${QUANT_MODEL} ];then
        echo "  "Quant Model Skip, ${QUANT_MODEL} has been existed
    else
        run_cmd python3 ${RUN_DIR}/quant.py        \
            --model ${SIM_MODEL}                   \
            --dataset_dir ${DATASET_DIR}/cityscapes           \
            --save_dir ${CHECKPOINTS_DIR}
        echo "  "Generate ${QUANT_MODEL}
    fi
    SIM_MODEL=${QUANT_MODEL}
    let step++
    echo;
fi

# Build Engine
echo [STEP ${step}] : Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.engine
if [ ${FORCE_BUILD} -eq 1 ] && [ -e ${ENGINE_FILE} ];then
    rm ${ENGINE_FILE}
fi
echo "Building engine(${PRECISION})"
if [ -e ${ENGINE_FILE} ];then
    echo "  "Build Engine Skip, ${ENGINE_FILE} has been existed
else
    run_cmd python3 ${RUN_DIR}/build_engine.py  \
        --model ${SIM_MODEL}                    \
        --bsz ${BSZ}                            \
        --precision ${PRECISION}                \
        --engine ${ENGINE_FILE}                 \
        --device ${DEVICE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi
let step++
echo;

# Inference
echo [STEP ${step}] : Inference
run_cmd python3 ${RUN_DIR}/inference.py     \
            --model_type "DDRNET23"         \
            --engine_file ${ENGINE_FILE}    \
            --test_mode MIOU                \
            --dataset_dir ${DATASET_DIR}    \
            --list_path ${LIST_PATH}        \
            --flip                          \
            --bsz ${BSZ}                    \
            --target_mIoU ${TGT_0}          \
            --target_mAcc ${TGT_1}          \
            --loop_count -1                 \
            --device ${DEVICE}; check_status

exit ${EXIT_STATUS}