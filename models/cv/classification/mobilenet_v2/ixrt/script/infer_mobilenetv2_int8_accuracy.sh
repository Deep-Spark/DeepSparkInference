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

# Run paraments
BSZ=32
TGT=-1
WARM_UP=0
LOOP_COUNT=-1
RUN_MODE=ACC
PRECISION=int8
IMGSIZE=224

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

RUN_DIR=${RUN_DIR}
PROJ_DIR=${PROJ_DIR}
DATASETS_DIR=${DATASETS_DIR}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR}

if [ ! -d $CHECKPOINTS_DIR ]; then
    mkdir -p $CHECKPOINTS_DIR
fi


MODEL_NAME="mobilenet_v2"
ORIGINE_MODEL="${CHECKPOINTS_DIR}/mobilenet_v2.onnx"

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo CONFIG_DIR : ${CONFIG_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}

step=0
# Export Onnx Model
let step++
echo;

SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
# Simplify Model
let step++
echo;
echo [STEP ${step}] : Simplify Model
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model, ${SIM_MODEL} has been existed
else
    python3 ${RUN_DIR}python/simplify_model.py  \
        --origin_model $ORIGINE_MODEL     \
        --output_model ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${BSZ}.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Change Batchsize Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}python/modify_batchsize.py  \
        --batch_size ${BSZ}                 \
        --origin_model ${SIM_MODEL}         \
        --output_model ${FINAL_MODEL}
    echo "  "Generate ${FINAL_MODEL}
fi

# Quantize Model
let step++
echo;
echo [STEP ${step}] : Quantize Model By PPQ
QUANTIZED_MODEL=${CHECKPOINTS_DIR}/quantized_${MODEL_NAME}.onnx
QUANTIZED_Q_PARAMS_JSON=${CHECKPOINTS_DIR}/quant_cfg.json
if [ -f $QUANTIZED_MODEL ];then
    echo "  "Quantized Model Skip By PPQ, $QUANTIZED_MODEL has been existed
else
    python3 ${RUN_DIR}python/quant.py           \
        --model_name ${MODEL_NAME}        \
        --model ${FINAL_MODEL}            \
        --dataset_dir ${DATASETS_DIR}     \
        --save_dir ${CHECKPOINTS_DIR}
    echo "  "Generate ${QUANTIZED_MODEL}
fi

# Build Engine
let step++
echo;
echo [STEP ${step}] : Build Engine By writing the ppq params onnx
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}python/build_engine_by_write_qparams.py          \
        --onnx ${QUANTIZED_MODEL}                                \
        --qparam_json ${QUANTIZED_Q_PARAMS_JSON}                 \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}python/inference.py     \
    --engine_file=${ENGINE_FILE}    \
    --datasets_dir=${DATASETS_DIR}  \
    --imgsz=${IMGSIZE}              \
    --warm_up=${WARM_UP}            \
    --loop_count ${LOOP_COUNT}      \
    --test_mode ${RUN_MODE}         \
    --acc_target ${TGT}             \
    --bsz ${BSZ}; check_status

exit ${EXIT_STATUS}
