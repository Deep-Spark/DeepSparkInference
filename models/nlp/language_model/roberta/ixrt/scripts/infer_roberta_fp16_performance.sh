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

    python3 ${PROJ_PATH}/gen_data.py --batch_size ${BS} --output_path ${PROJ_PATH}

    # Graph optimize
    [ -f "${TARGET_ONNX}" ] || python3 ${OPTIMIER_FILE} --onnx ${ORIGIN_ONNX} --dump_onnx
    
    # Build Engine
    ixrtexec --onnx ${TARGET_ONNX} --min_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                   --opt_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                   --max_shape input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384 \
                                   --save_engine ${TARGET_ENGINE} --log_level error --plugins ixrt_plugin

    # Test Performance
    ixrtexec --load_engine ${TARGET_ENGINE} --plugins ixrt_plugin --shapes input_ids.1:${BS}x384,attention_mask.1:${BS}x384,token_type_ids.1:${BS}x384

}
run 1