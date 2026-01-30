#!/bin/bash

# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    echo "fails"
    EXIT_STATUS=1
    fi
}
# Run paraments
warm_up=10
BSZ=32
TGT=-1

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

datasets_dir=${DATASETS_DIR}
onnx_model=${CHECKPOINTS_DIR}/swin_s_model_sim.onnx
engine_file=${CHECKPOINTS_DIR}/swin_s_model_sim.engine

echo "Build Fp16 Engine!"
if [ -f ${engine_file} ];then
  echo "  "Build Engine Skip, $engine_file has been existed
else
  python3 ${RUN_DIR}build_engine.py             \
          --precision float16         \
          --model ${onnx_model}       \
          --engine ${engine_file}; check_status
fi

echo "Fp16 Inference Acc!"
python3 ${RUN_DIR}inference.py                        \
        --test_mode ACC                     \
        --engine_file ${engine_file}        \
        --datasets_dir ${datasets_dir}      \
        --warm_up ${warm_up}                \
        --bsz ${BSZ}                        \
        --acc_target ${TGT};  check_status
