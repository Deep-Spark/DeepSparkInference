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

PROJ_DIR=$(cd $(dirname $0);cd ../; pwd)
DATASETS_DIR="${PROJ_DIR}/data/coco"
COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
EVAL_DIR=${DATASETS_DIR}/images/val2017
CHECKPOINTS_DIR="${PROJ_DIR}/data"
RUN_DIR="${PROJ_DIR}"
ORIGINE_MODEL=${CHECKPOINTS_DIR}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo ====================== Model Info ======================
echo Model Name : yolov4_darknet
echo Onnx Path : ${ORIGINE_MODEL}

BATCH_SIZE=16
CURRENT_MODEL=${CHECKPOINTS_DIR}/yolov4_sim.onnx

# Cut decoder part
echo "Cut decoder part"
FINAL_MODEL=${CHECKPOINTS_DIR}/yolov4_bs${BATCH_SIZE}_without_decoder.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "CUT Model Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/cut_model.py  \
        --input_model ${CURRENT_MODEL}     \
        --output_model ${FINAL_MODEL}      \
        --input_names input                \
        --output_names /models.138/conv94/Conv_output_0 /models.149/conv102/Conv_output_0 /models.160/conv110/Conv_output_0
    echo "  "Generate ${FINAL_MODEL}
fi
CURRENT_MODEL=${FINAL_MODEL}

# add decoder op
FINAL_MODEL=${CHECKPOINTS_DIR}/yolov4_bs${BATCH_SIZE}_with_decoder.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Add Decoder Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/deploy.py             \
        --src ${CURRENT_MODEL}               \
        --dst ${FINAL_MODEL}                 \
        --decoder_type YoloV3Decoder          \
        --decoder_input_names /models.138/conv94/Conv_output_0 /models.149/conv102/Conv_output_0 /models.160/conv110/Conv_output_0 \
        --decoder8_anchor 12 16 19 36 40 28            \
        --decoder16_anchor 36 75 76 55 72 146          \
        --decoder32_anchor 142 110 192 243 459 401
    echo "  "Generate ${FINAL_MODEL}
fi
CURRENT_MODEL=${FINAL_MODEL}

# Build Engine
echo Build Engine
ENGINE_FILE=${CHECKPOINTS_DIR}/yolov4_fp16.engine
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
RUN_BATCH_SIZE=16
python3 ${RUN_DIR}/inference.py                 \
    --test_mode FPS                             \
    --model_engine ${ENGINE_FILE}                \
    --warm_up 2                                 \
    --bsz ${RUN_BATCH_SIZE}                         \
    --imgsz 608                              \
    --loop_count 10                             \
    --eval_dir ${EVAL_DIR}                      \
    --coco_gt ${COCO_GT}                        \
    --pred_dir ${CHECKPOINTS_DIR}               \
    --precision float16                            \
    --map_target 0.30; check_status
exit ${EXIT_STATUS}
