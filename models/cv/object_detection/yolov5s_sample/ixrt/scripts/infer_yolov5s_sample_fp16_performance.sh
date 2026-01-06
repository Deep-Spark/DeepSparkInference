#!/bin/bash

EXIT_STATUS=0
check_status()
{
    ret_code=${PIPESTATUS[0]}
    if [ ${ret_code} != 0 ]; then
    [[ ${ret_code} -eq 10 && "${TEST_PERF:-1}" -eq 0 ]] || EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=32
WARM_UP=3
TGT=840
LOOP_COUNT=100
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
DATASETS_DIR="${PROJ_DIR}/coco"
COCO_GT=${DATASETS_DIR}/annotations/instances_val2017.json
EVAL_DIR=${DATASETS_DIR}/images/val2017
CHECKPOINTS_DIR="${PROJ_DIR}/checkpoints"
RUN_DIR="${PROJ_DIR}"
CONFIG_DIR="${RUN_DIR}/config/YOLOV5S_CONFIG"
source ${CONFIG_DIR}
ORIGINE_MODEL=${CHECKPOINTS_DIR}/${ORIGINE_MODEL}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo CONFIG_DIR : ${CONFIG_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Onnx Path : ${ORIGINE_MODEL}

CHECKPOINTS_DIR=${CHECKPOINTS_DIR}/tmp
mkdir -p ${CHECKPOINTS_DIR}

step=0
faster=0
CURRENT_MODEL=${ORIGINE_MODEL}
if [[ ${LAYER_FUSION} == 1 && ${DECODER_FASTER} == 1 ]];then
    faster=1
fi

# Simplify Model
let step++
echo [STEP ${step}] : Simplify Model
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_sim.onnx
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model skip, ${SIM_MODEL} has been existed
else
    python3 ${RUN_DIR}/simplify_model.py    \
    --origin_model ${CURRENT_MODEL}         \
    --output_model ${SIM_MODEL}
    echo "  "Generate ${SIM_MODEL}
fi
CURRENT_MODEL=${SIM_MODEL}

# Cut Decoder
let step++
echo [STEP ${step}] : Cut Decoder
NO_DECODER_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_without_decoder.onnx
if [ -f ${NO_DECODER_MODEL} ];then
    echo "  "Cut Decoder skip, ${SIM_MNO_DECODER_MODELODEL} has been existed
else
    python3 ${RUN_DIR}/cut_model.py         \
    --input_model  ${CURRENT_MODEL}         \
    --output_model ${NO_DECODER_MODEL}      \
    --input_names ${MODEL_INPUT_NAMES[@]}   \
    --output_names ${DECODER_INPUT_NAMES[@]}
fi
CURRENT_MODEL=${NO_DECODER_MODEL}

# Quant Model
if [ $PRECISION == "int8" ];then
    let step++
    echo;
    echo [STEP ${step}] : Quant Model
    if [[ -z ${QUANT_EXIST_ONNX} ]];then
        QUANT_EXIST_ONNX=$CHECKPOINTS_DIR/quantized_${MODEL_NAME}.onnx
    fi
    if [[ -f ${QUANT_EXIST_ONNX} ]];then
        CURRENT_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Quant Model Skip, ${QUANT_EXIST_ONNX} has been existed
    else
        python3 ${RUN_DIR}/quant.py                         \
            --model ${CURRENT_MODEL}                        \
            --model_name ${MODEL_NAME}                      \
            --dataset_dir ${EVAL_DIR}                       \
            --ann_file ${COCO_GT}                           \
            --data_process_type ${DATA_PROCESS_TYPE}        \
            --observer ${QUANT_OBSERVER}                    \
            --disable_quant_names ${DISABLE_QUANT_LIST[@]}  \
            --save_dir $CHECKPOINTS_DIR                     \
            --bsz   ${QUANT_BATCHSIZE}                      \
            --step  ${QUANT_STEP}                           \
            --seed  ${QUANT_SEED}                           \
            --imgsz ${IMGSIZE}
        echo "  "Generate ${QUANT_EXIST_ONNX}
    fi
    CURRENT_MODEL=${QUANT_EXIST_ONNX}
fi

# Add Decoder
if [ $LAYER_FUSION == "1" ]; then
    let step++
    echo;
    echo [STEP ${step}] : Add Decoder
    FUSION_ONNX=${CHECKPOINTS_DIR}/${MODEL_NAME}_fusion_no_cancat.onnx
    if [ -f $FUSION_ONNX ];then
        echo "  "Add Decoder Skip, $FUSION_ONNX has been existed
    else
        python3 ${RUN_DIR}/deploy.py                        \
            --src ${CURRENT_MODEL}                          \
            --dst ${FUSION_ONNX}                            \
            --decoder_type        YoloV5Decoder             \
            --with_nms             False                    \
            --decoder_input_names ${DECODER_INPUT_NAMES[@]} \
            --decoder8_anchor     ${DECODER_8_ANCHOR[@]}    \
            --decoder16_anchor    ${DECODER_16_ANCHOR[@]}   \
            --decoder32_anchor    ${DECODER_32_ANCHOR[@]}   \
            --num_class           ${DECODER_NUM_CLASS}      \
            --faster              ${faster}
    fi
    CURRENT_MODEL=${FUSION_ONNX}
fi

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_bs${BSZ}_without_nms.onnx
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
ENGINE_FILE=${CHECKPOINTS_DIR}/${MODEL_NAME}_${PRECISION}_bs${BSZ}_without_nms.engine
if [ -f $ENGINE_FILE ];then
    echo "  "Build Engine Skip, $ENGINE_FILE has been existed
else
    python3 ${RUN_DIR}/build_engine.py          \
        --precision ${PRECISION}                \
        --model ${CURRENT_MODEL}                \
        --engine ${ENGINE_FILE}
    echo "  "Generate Engine ${ENGINE_FILE}
fi
if [[ ${RUN_MODE} == "MAP" && ${NMS_TYPE} == "GPU" ]];then
    NMS_ENGINE=${CHECKPOINTS_DIR}/nms.engine
    # Build NMS Engine
    python3 ${RUN_DIR}/build_nms_engine.py      \
        --bsz ${BSZ}                            \
        --path ${CHECKPOINTS_DIR}               \
        --all_box_num ${ALL_BOX_NUM}            \
        --max_box_pre_img   ${MAX_BOX_PRE_IMG}  \
        --iou_thresh        ${IOU_THRESH}       \
        --score_thresh      ${SCORE_THRESH}
fi

# Inference
let step++
echo;
echo [STEP ${step}] : Inference
python3 ${RUN_DIR}/inference.py                 \
    --model_engine=${ENGINE_FILE}               \
    --nms_engine=${NMS_ENGINE}                  \
    --coco_gt=${COCO_GT}                        \
    --eval_dir=${EVAL_DIR}                      \
    --data_process_type ${DATA_PROCESS_TYPE}    \
    --decoder_faster=${faster}                  \
    --imgsz=${IMGSIZE}                          \
    --warm_up=${WARM_UP}                        \
    --loop_count ${LOOP_COUNT}                  \
    --test_mode ${RUN_MODE}                     \
    --model_name ${MODEL_NAME}                  \
    --precision  ${PRECISION}                   \
    --pred_dir   ${CHECKPOINTS_DIR}             \
    --fps_target ${TGT}                         \
    --max_det ${MAX_BOX_PRE_IMG}                \
    --nms_type ${NMS_TYPE}                      \
    --bsz ${BSZ}; check_status
exit ${EXIT_STATUS}