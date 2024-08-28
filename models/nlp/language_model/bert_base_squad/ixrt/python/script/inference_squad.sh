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

BSZ=1
USE_FP16=True

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --int8) USE_FP16=False;;
    esac
done

if [ "$USE_FP16" = "True" ]; then
    echo 'USE_FP16=True'
    UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ./data/bert_base_384.engine \
                            -b ${BSZ} \
                            -s 384 \
                            -sq ./data/squad/dev-v1.1.json \
                            -v ./data/bert_base_uncased_squad/vocab.txt \
                            -o ./data/predictions-bert_base_384.json 
    python3 evaluate-v1.1.py  ./data/squad/dev-v1.1.json  ./data/predictions-bert_base_384.json 87
else
    echo 'USE_INT8=True'
    UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ./data/bert_base_384_int8.engine \
                            -b ${BSZ} \
                            -s 384 \
                            -sq ./data/squad/dev-v1.1.json \
                            -v ./data/bert_base_uncased_squad/vocab.txt \
                            -o ./data/predictions-bert_base_384_int8.json \
                            -i
    python3 evaluate-v1.1.py  ./data/squad/dev-v1.1.json  ./data/predictions-bert_base_384_int8.json 86
fi