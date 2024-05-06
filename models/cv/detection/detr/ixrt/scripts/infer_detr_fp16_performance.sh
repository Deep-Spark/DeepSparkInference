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

#!/bin/bash

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=1
WARM_UP=3
TGT=-1
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
faster=0
CURRENT_MODEL=${ORIGINE_MODEL}
if [[ ${LAYER_FUSION} == 1 && ${DECODER_FASTER} == 1 ]];then
    faster=1
fi

# Simplify Model
let step++
echo;
echo [STEP ${step}] : Simplify Model
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model Skipped, ${SIM_MODEL} has been existed
else
    python3 ${RUN_DIR}/simplify_model.py \
            --origin_model $ORIGINE_MODEL    \
            --output_model ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi

# Refine Model
let step++
echo;
echo [STEP ${step}] : Refine Model
REFINE_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_refine.onnx
if [ -f ${REFINE_MODEL} ];then
    echo "  "Refine Model Skipped, ${REFINE_MODEL} has been existed
else
    python3 ${RUN_DIR}/refine_model.py \
            --onnx_path  ${SIM_MODEL} \
            --dst_onnx_path ${REFINE_MODEL} \
            --bsz  ${BSZ}               \
            --imgsz  ${IMGSIZE}
fi

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Change Batchsize Skipped, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/modify_batchsize.py \
            --batch_size ${BSZ} \
            --origin_model ${REFINE_MODEL} \
            --output_model ${FINAL_MODEL}
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
        --model ${FINAL_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference.py                 \
    --model_engine=${ENGINE_FILE}               \
    --coco_gt=${COCO_GT}                        \
    --eval_dir=${EVAL_DIR}                      \
    --data_process_type ${DATA_PROCESS_TYPE}    \
    --imgsz=${IMGSIZE}                          \
    --warm_up=${WARM_UP}                        \
    --loop_count ${LOOP_COUNT}                  \
    --test_mode ${RUN_MODE}                     \
    --model_name ${MODEL_NAME}                  \
    --precision  ${PRECISION}                   \
    --pred_dir   ${CHECKPOINTS_DIR}             \
    --fps_target ${TGT}                         \
    --bsz ${BSZ}; check_status
exit ${EXIT_STATUS}