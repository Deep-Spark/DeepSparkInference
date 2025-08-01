#!/bin/bash

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    EXIT_STATUS=1
    fi
}

# Run paraments
BSZ=64
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

PROJ_DIR=$(cd $(dirname $0);cd ../../; pwd)
echo PROJ_DIR : ${PROJ_DIR}
RUN_DIR="${PROJ_DIR}/ixrt/"
DATASETS_DIR="${RUN_DIR}/facenet_datasets/"
CHECKPOINTS_DIR="${RUN_DIR}/facenet_weights/"
CONFIG_DIR="${PROJ_DIR}/ixrt/config/FACENET_CONFIG"
source ${CONFIG_DIR}
ORIGINE_MODEL=${CHECKPOINTS_DIR}/${ORIGINE_MODEL}

echo CHECKPOINTS_DIR : ${CHECKPOINTS_DIR}
echo DATASETS_DIR : ${DATASETS_DIR}
echo RUN_DIR : ${RUN_DIR}
echo CONFIG_DIR : ${CONFIG_DIR}
echo ====================== Model Info ======================
echo Model Name : ${MODEL_NAME}
echo Model Input Name : ${MODEL_INPUT_NAME}
echo Model Output Name : ${MODEL_OUTPUT_NAME}
echo Onnx Path : ${ORIGINE_MODEL}

step=0
SIM_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}.onnx

# Simplify Model
let step++
echo;
echo [STEP ${step}] : Simplify Model
if [ -f ${SIM_MODEL} ];then
    echo "  "Simplify Model, ${SIM_MODEL} has been existed
else
    cd $RUN_DIR
    python3 ${RUN_DIR}/deploy.py \
    --onnx_name ${CHECKPOINTS_DIR}/facenet_export.onnx
    echo "  "Generate ${SIM_MODEL}
fi

# Quant Model
if [ $PRECISION == "int8" ];then
    let step++
    echo;
    echo [STEP ${step}] : Quant Model
    if [[ -z ${QUANT_EXIST_ONNX} ]];then
        QUANT_EXIST_ONNX=$CHECKPOINTS_DIR/${MODEL_NAME}-quant.onnx
    fi
    if [[ -f ${QUANT_EXIST_ONNX} ]];then
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Quant Model Skip, ${QUANT_EXIST_ONNX} has been existed
    else
        python3 ${RUN_DIR}/quant.py            \
            --model ${SIM_MODEL} \
            --batch_size ${QUANT_BATCHSIZE} \
            --img_size ${IMGSIZE} \
            --num_samples 6400 \
            --observer ${QUANT_OBSERVER} \
            --disable_bias_correction
        SIM_MODEL=${QUANT_EXIST_ONNX}
        echo "  "Generate ${SIM_MODEL}
    fi
fi

# Change Batchsize
let step++
echo;
echo [STEP ${step}] : Change Batchsize
FINAL_MODEL=${CHECKPOINTS_DIR}/${MODEL_NAME}_${BSZ}.onnx
if [ -f $FINAL_MODEL ];then
    echo "  "Change Batchsize Skip, $FINAL_MODEL has been existed
else
    python3 ${RUN_DIR}/modify_batchsize.py --batch_size ${BSZ} \
        --origin_model ${SIM_MODEL} --output_model ${FINAL_MODEL}
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