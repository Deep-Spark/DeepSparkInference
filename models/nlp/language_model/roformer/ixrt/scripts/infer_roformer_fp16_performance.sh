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
set -x
ORIGIN_ONNX=${ORIGIN_ONNX_NAME}.onnx
cd ${PROJ_PATH}

run(){
    BS=${1:-1}
    TARGET_ONNX=${ORIGIN_ONNX_NAME}_end.onnx
    TARGET_ENGINE=${ORIGIN_ONNX_NAME}_end.engine
    SHAPE="input_segment0:${BS}x1024,input_token0:${BS}x1024"
    MAX_SHAPE="input_segment0:64x1024,input_token0:64x1024"
    if [[ ! -f "${ORIGIN_ONNX}" ]];then
        echo "${ORIGIN_ONNX} not exists!"
        exit 1
    fi

    # Graph optimize
    python3 ${OPTIMIER_FILE} --onnx ${ORIGIN_ONNX} --model_type roformer

    # Build Engine
    ixrtexec --onnx ${TARGET_ONNX} --save_engine ${TARGET_ENGINE} --log_level error --plugins ixrt_plugin \
        --min_shape $SHAPE --opt_shape $SHAPE --max_shape $MAX_SHAPE --shapes $SHAPE

    # Test Performance
    ixrtexec --load_engine ${TARGET_ENGINE} --plugins ixrt_plugin --shapes ${SHAPE}

}
run 2