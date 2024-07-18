set -x
ORIGIN_ONNX=${ORIGIN_ONNX_NAME}.onnx
cd ${PROJ_PATH}

run(){
    BS=${1:-1}
    TARGET_ONNX=${ORIGIN_ONNX_NAME}_end.onnx
    TARGET_ENGINE=${ORIGIN_ONNX_NAME}_bs_${BS}_end.engine
    if [[ ! -f "${ORIGIN_ONNX}" ]];then
        echo "${ORIGIN_ONNX} not exists!"
        exit 1
    fi

    # Graph optimize
    python3 ${OPTIMIER_FILE} --onnx ${ORIGIN_ONNX} --input_shapes "new_categorical_placeholder0:$((26 * ${BS}))x2,new_numeric_placeholder0:${BS}x13,import/head/predictions/zeros_like0:${BS}x1"
    # Build Engine
    ixrtexec --onnx ${TARGET_ONNX} --save_engine ${TARGET_ENGINE} --log_level error

    # Test Performance
    ixrtexec --load_engine ${TARGET_ENGINE}

}
run 1