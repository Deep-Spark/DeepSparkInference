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

DATASETS_DIR=${DATASETS_DIR}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR}

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
        EXIT_STATUS=1
    fi
}

python3 build_engine.py --model_name lightweight_openpose \
    --onnx_path ${CHECKPOINTS_DIR}/lightweight_openpose.onnx \
    --engine_path ${CHECKPOINTS_DIR}/lightweight_openpose.engine \
    --engine_path_dynamicshape ${CHECKPOINTS_DIR}/lightweight_openpose_dynamicshape.engine

python3 inference_performance.py \
    --model_type lightweight_openpose \
    --engine_file ${CHECKPOINTS_DIR}/lightweight_openpose.engine \
    --datasets_dir ${DATASETS_DIR} \
    --bsz 32 \
    --imgh 256 \
    --imgw 456 \
    --test_mode FPS \
    --warm_up 10 \
    --run_loop 20 \
    --device 0  "$@";check_status

exit ${EXIT_STATUS}
