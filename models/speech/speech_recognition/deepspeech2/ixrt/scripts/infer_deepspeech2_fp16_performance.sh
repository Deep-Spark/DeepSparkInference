#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

DATASETS_DIR="${DATASETS_DIR:-/path/to/LibriSpeech}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
RUN_DIR="${RUN_DIR:-.}"

TGT=-1
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --tgt) TGT=${arguments[index]};;
    esac
done

cd ${RUN_DIR}
python3 inference_demo.py \
        --model_type "deepspeech2" \
        --engine_file "${CHECKPOINTS_DIR}/deepspeech2.engine" \
        --decoder_file "data/decoder.pdparams" \
        --lang_model_path "${CHECKPOINTS_DIR}/common_crawl_00.prune01111.trie.klm" \
        --run_loop 12 \
        --warm_up 5 \
        --throughput_target ${TGT}