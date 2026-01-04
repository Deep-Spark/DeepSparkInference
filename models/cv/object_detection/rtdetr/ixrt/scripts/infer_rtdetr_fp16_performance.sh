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
BSZ=8
WARM_UP=-1
TGT=-1
LOOP_COUNT=-1
RUN_MODE=FPS
PRECISION=float16
DATA_PROCESS_TYPE=rtdetrv3

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
DATASETS_DIR=${DATASETS_DIR}
COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
EVAL_DIR=${DATASETS_DIR}/val2017
CHECKPOINTS_DIR="${PROJ_DIR}/checkpoints"
RUN_DIR="${PROJ_DIR}"
ORIGINE_MODEL=${CHECKPOINTS_DIR}/rtdetrv3_r18vd_6x_coco_image_sim.onnx

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo ====================== Model Info ======================
echo Model Name : rt-detr-v3
echo Onnx Path : ${ORIGINE_MODEL}

CURRENT_MODEL=${CHECKPOINTS_DIR}/rtdetrv3_r18vd_6x_coco_image_sim.onnx

# Build Engine
echo Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/rtdetrv3_fp16.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_dynamic_engine.py          \
        --precision float16                     \
        --model ${CURRENT_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
echo Inference
python3 ${RUN_DIR}/inference.py                 \
    --model_engine ${ENGINE_FILE}               \
    --coco_gt=${COCO_GT}                        \
    --eval_dir=${EVAL_DIR}                      \
    --data_process_type ${DATA_PROCESS_TYPE}    \
    --precision  ${PRECISION}                   \
    --imgsz 640                                 \
    --test_mode ${RUN_MODE}                     \
    --fps_target ${TGT}                         \
    --bsz ${BSZ}; check_status               
exit ${EXIT_STATUS}