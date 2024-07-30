#!/bin/bash
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

index=0
options=("$@") # 将所有参数存储到数组中
PRECISION=fp16
BSZ=32

# 循环遍历所有参数
while [[ $index -lt ${#options[@]} ]]; do
    argument=${options[$index]}
    case $argument in
    --bs)
        ((index++))
        BSZ=${options[$index]}
        ;;
    --prec)
        ((index++))
        PRECISION=${options[$index]}
        ;;
    esac
    ((index++))
done

# 设置INT8_FLAG
INT8_FLAG=""
if [[ "$PRECISION" == "int8" ]]; then
    INT8_FLAG="--int8"
fi

# 设置BSZ_FLAG
BSZ_FLAG=""
if [[ "$BSZ" -ne 32 ]]; then
    BSZ_FLAG="--bs $BSZ"
fi

echo "PREC_FLAG=$INT8_FLAG"
echo "PRECISION=$PRECISION"
echo "BSZ=$BSZ"
echo "BSZ_FLAG=$BSZ_FLAG"

# 检查环境并执行相应的脚本
if command -v ixsmi &>/dev/null; then
    echo "MR env"
    cmake -S . -B build
    cmake --build build -j16
    cd ./python/script/
    bash infer_bert_base_squad_${PRECISION}_ixrt.sh $BSZ_FLAG

elif command -v nvidia-smi &>/dev/null; then
    echo "NV env"
    cmake -S . -B build -DUSE_TENSORRT=true
    cmake --build build -j16
    cd ./python/
    bash script/build_engine.sh --bs $BSZ $INT8_FLAG
    bash script/inference_squad.sh --bs $BSZ $INT8_FLAG
else
    echo "No driver detected"
    exit 1
fi
