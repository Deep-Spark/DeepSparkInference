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
set -eo pipefail

BSZ=32
TGT=86
USE_TRT=False

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
      --use_trt) USE_TRT=${arguments[index]};;
    esac
done

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
project_path=$(realpath ${current_path}/..)
echo ${project_path}
checkpoints_path=${project_path}/data/bert_base_uncased_squad/
datasets_path=${project_path}/data/

echo 'USE_TRT='${USE_TRT}
export USE_TRT=$USE_TRT

echo "Step1 Build Engine Int8(bert base squad)!"
cd ${project_path}/ixrt
python3 builder_int8.py -pt ${checkpoints_path}/bert_base_int8_qat.bin \
                -o ${checkpoints_path}/bert_base_int8_b${BSZ}.engine \
                -b 1 ${BSZ} ${BSZ} \
                -s 1 384 384 \
                -i \
                -c ${checkpoints_path}

echo "Step2 Run dev.json and generate json"
python3 inference.py -e ${checkpoints_path}/bert_base_int8_b${BSZ}.engine \
                        -b ${BSZ} \
                        -s 384 \
                        -sq ${datasets_path}/squad/dev-v1.1.json \
                        -v ${checkpoints_path}/vocab.txt \
                        -o ${checkpoints_path}/predictions-bert_base_int8_b${BSZ}.json \
                        -z ${USE_TRT} \
                        -i

echo "Step3 Inference(test F1-score)"
python3 evaluate-v1.1.py  ${datasets_path}/squad/dev-v1.1.json  ${checkpoints_path}/predictions-bert_base_int8_b${BSZ}.json ${TGT}