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
set -euo pipefail
batchsize=24

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) batchsize=${arguments[index]};;
    esac
done

echo "batch size is ${batchsize}"

EXIT_STATUS=0
check_status()
{
    ret_code=${PIPESTATUS[0]}
    if [ ${ret_code} != 0 ]; then
    echo "fails"
    [[ ${ret_code} -eq 10 && "${TEST_PERF:-1}" -eq 0 ]] || EXIT_STATUS=1
    fi
}

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

PROJECT_DIR=${current_path}/..
DATA_DIR=${current_path}/../aishell_test_data/test
MODEL_DIR=${current_path}/../conformer_checkpoints

export Accuracy=${Accuracy:=0.05}

cd ${PROJECT_DIR}

python3 build_engine.py \
        --onnx_model ${MODEL_DIR}/conformer_fp16_trt.onnx  \
        --engine ${MODEL_DIR}/conformer_fp16_trt.engine;check_status 

python3 ixrt_inference_accuracy.py \
    --infer_type fp16 \
    --batch_size ${batchsize} \
    --data_dir ${DATA_DIR}  \
    --model_dir ${MODEL_DIR}; check_status
exit ${EXIT_STATUS}