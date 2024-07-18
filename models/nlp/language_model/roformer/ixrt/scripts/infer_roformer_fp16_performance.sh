set -x
ORIGIN_ONNX=${ORIGIN_ONNX_NAME}.onnx
cd ${PROJ_PATH}

run(){
    BS=${1:-1}
    TARGET_ONNX=${ORIGIN_ONNX_NAME}_end.onnx
    TARGET_ENGINE=${ORIGIN_ONNX_NAME}_bs_${BS}_end.engine
    SHAPE="input_segment0:${BS}x1024,input_token0:${BS}x1024"
    if [[ ! -f "${ORIGIN_ONNX}" ]];then
        echo "${ORIGIN_ONNX} not exists!"
        exit 1
    fi

    # Graph optimize
    python3 ${OPTIMIER_FILE} --onnx ${ORIGIN_ONNX} --model_type roformer

    # Build Engine
    ixrtexec --onnx ${TARGET_ONNX} --save_engine ${TARGET_ENGINE} --log_level error --plugins ixrt_plugin \
        --min_shape $SHAPE --opt_shape $SHAPE --max_shape $SHAPE --shapes $SHAPE

    # Test Performance
    ixrtexec --load_engine ${TARGET_ENGINE} --plugins ixrt_plugin --shapes ${SHAPE}

}
run 1