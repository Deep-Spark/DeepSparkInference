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


# Start to test
set -x
ORIGIN_ONNX=${ORIGIN_ONNX_NAME}.onnx
cd ${PROJ_PATH}

run(){
    BS=16
    TARGET_ONNX=${ORIGIN_ONNX_NAME}_end.onnx
    TARGET_ENGINE=${ORIGIN_ONNX_NAME}_end.engine
    if [[ ! -f "${ORIGIN_ONNX}" ]];then
        echo "${ORIGIN_ONNX} not exists!"
        exit 1
    fi

    # Graph optimize
    python3 ${OPTIMIER_FILE} --onnx ${ORIGIN_ONNX} --dump_onnx

    # Build Engine
    ixrtexec --onnx ${TARGET_ONNX} --min_shape image:${BS}x3x224x224,text:100x77 \
                                   --opt_shape image:${BS}x3x224x224,text:100x77 \
                                   --max_shape image:${BS}x3x224x224,text:100x77 \
                                   --save_engine ${TARGET_ENGINE} --log_level error --plugins ixrt_plugin --shapes image:${BS}x3x224x224,text:100x77

    # Test Performance
    ixrtexec --load_engine ${TARGET_ENGINE} --plugins ixrt_plugin --shapes image:${BS}x3x224x224,text:100x77

}
run 1