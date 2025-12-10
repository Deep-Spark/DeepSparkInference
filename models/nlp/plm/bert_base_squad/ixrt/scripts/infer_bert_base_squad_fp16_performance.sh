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
set -eo pipefail

BSZ=32
TGT=900
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

project_path=./
checkpoints_path=${project_path}/data/checkpoints/bert_base_squad/bert_base_uncased_squad
datasets_path=${project_path}/data/datasets/bert_base_squad/

echo 'USE_TRT='${USE_TRT}
export USE_TRT=$USE_TRT

echo "Step1 Build Engine FP16(bert base squad)!"
python3 builder.py -x ${checkpoints_path}/bert_base_squad.onnx \
                   -w 4096 \
                   -o ${checkpoints_path}/bert_base_b${BSZ}.engine \
                   -s 1 384 384 \
                   -b 1 ${BSZ} ${BSZ} \
                   --fp16 \
                   -c ${checkpoints_path}/config.json \
                   -z ${USE_TRT}

echo "Step2 Inference(test QPS)"
UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ${checkpoints_path}/bert_base_b${BSZ}.engine \
                        -s 384 \
                        -b ${BSZ} \
                        -sq ${datasets_path}/squad/dev-v1.1.json \
                        -v ${checkpoints_path}/vocab.txt \
                        -o ${checkpoints_path}/predictions-bert_base_b${BSZ}.json \
                        -z ${USE_TRT} \
                        --target_qps ${TGT}