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

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

PROJ_DIR=${PROJ_DIR}
DATASETS_DIR=${DATASETS_DIR}
COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
EVAL_DIR=${DATASETS_DIR}/images/val2017
CHECKPOINTS_DIR=${CHECKPOINTS_DIR}
RUN_DIR=${PROJ_DIR}
ORIGINE_MODEL=${CHECKPOINTS_DIR}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo ====================== Model Info ======================
echo Model Name : yolov8
echo Onnx Path : ${ORIGINE_MODEL}

BATCH_SIZE=32
CURRENT_MODEL=${CHECKPOINTS_DIR}/yolov8.onnx

# Build Engine
echo Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/yolov8_fp16.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --precision float16                        \
        --model ${CURRENT_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
echo Inference
RUN_BATCH_SIZE=32
python3 ${RUN_DIR}/inference.py                 \
    --model_engine ${ENGINE_FILE}                \
    --warm_up 2                                 \
    --bsz ${RUN_BATCH_SIZE}                         \
    --imgsz 640                              \
    --datasets ${DATASETS_DIR}               \
    --perf_only true                         \
    --fps_target 0.0                     
exit ${EXIT_STATUS}
