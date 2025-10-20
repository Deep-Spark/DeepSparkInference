#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=32
WARM_UP=-1
TGT=-1
LOOP_COUNT=-1
RUN_MODE=MAP
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

MODEL_NAME="yolof"
ORIGINE_MODEL="${CHECKPOINTS_DIR}/${MODEL_NAME}.onnx"

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}

step=0
CURRENT_MODEL=${ORIGINE_MODEL}
# Simplify Model
let step++
echo;
echo [STEP ${step}] : Simplify Model
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model Skipped, ${SIM_MODEL} has been existed
else
    python3 ${RUN_DIR}/simplify_model.py \
            --origin_model ${CURRENT_MODEL}   \
            --output_model ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi

CURRENT_MODEL=${SIM_MODEL}

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_bs${BSZ}.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Change Batchsize Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/modify_batchsize.py  \
        --batch_size ${BSZ}                 \
        --origin_model ${CURRENT_MODEL}     \
        --output_model ${FINAL_MODEL}
    echo "  "Generate ${FINAL_MODEL}
fi
CURRENT_MODEL=${FINAL_MODEL}


# Build Engine
let step++
echo;
echo [STEP ${step}] : Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --model ${CURRENT_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference_mmdet.py \
        --engine ${ENGINE_FILE} \
        --cfg_file ${RUN_DIR}/yolof_r50-c5_8xb8-1x_coco.py \
        --perf_only True \
        --datasets ${DATASETS_DIR} \
        --batchsize ${BSZ} \
        --fps_target ${TGT}; check_status
exit ${EXIT_STATUS}
