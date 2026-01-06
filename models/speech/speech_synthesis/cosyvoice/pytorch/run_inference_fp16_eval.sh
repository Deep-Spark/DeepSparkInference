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

SPK_LAB=utils/3D-Speaker
DNSMOS_LAB=utils/DNSMOS

export PYTHONPATH=${SPK_LAB}:${PYTHONPATH}

nj=1
asr_gpu=1

task=zero_shot
dumpdir=data/${task}
# test_set="zh en hard_zh hard_en"
test_set="zh"

# task=cross_lingual_zeroshot
# dumpdir=data/${task}
# # test_set="to_zh to_en to_hard_zh to_hard_en to_ja to_ko" 
# test_set="to_zh to_en" 

# task=emotion_zeroshot
# dumpdir=data/${task}
# test_set="en zh" 

# task=subjective_zeroshot
# dumpdir=data/${task}

# task=subjective_continue
# dumpdir=data/${task}
# test_set="emotion rhyme speed volume"

inference_dir=$(pwd)
. utils/parse_options.sh || exit 1;
inference_tag="${inference_dir}/${task}"
decode_dir=${inference_dir}/results/${task}

for lang in ${test_set}; do
    name_without_extension=$lang

    echo "Do inference..."
    bash scripts/inference.sh ${inference_dir} ${dumpdir}/${name_without_extension} ${decode_dir}/${name_without_extension}/wavs

    echo "Score WER for ${decode_dir}/${name_without_extension}"

    bash scripts/run_score_wer.sh ${dumpdir}/${name_without_extension}/text ${decode_dir}/${name_without_extension} ${name_without_extension} ${asr_gpu}

    find ${decode_dir}/${name_without_extension}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${name_without_extension}/wav.scp

    echo "Score 3DSpeaker for ${decode_dir}/${name_without_extension}"
    python3 scripts/eval_speaker_similarity.py \
    --model_id damo/speech_eres2net_sv_en_voxceleb_16k \
    --local_model_dir ${SPK_LAB}/pretrained \
    --prompt_wavs ${dumpdir}/${name_without_extension}/prompt_wav.scp \
    --hyp_wavs ${decode_dir}/${name_without_extension}/wav.scp \
    --log_file ${decode_dir}/${name_without_extension}/spk_simi_scores.txt \
    --devices "0"

    echo "Score DNSMOS  for ${decode_dir}/${name_without_extension}"
    python3 ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${name_without_extension}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${name_without_extension}/mos.csv

    cat ${decode_dir}/${name_without_extension}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${name_without_extension}/dnsmos_mean.txt


    spk_simi_file="${decode_dir}/${name_without_extension}/spk_simi_scores.txt"
    if [ ! -f "$spk_simi_file" ]; then
        echo "❌ Error: File not found: $spk_simi_file"
        exit 1
    fi
    spk_simi_score=$(awk '/^avg[[:space:]]/ { print $NF }' "$spk_simi_file")
    if [ -z "$spk_simi_score" ]; then
        echo "❌ Error: No 'avg' line found in $spk_simi_file"
        exit 1
    fi

    wer_file="${decode_dir}/${name_without_extension}/wav_res_ref_text.wer"
    if [ ! -f "$wer_file" ]; then
        echo "❌ Error: File not found: $wer_file"
        exit 1
    fi
    wer_score=$(grep -oP '^WER:\s*\K[0-9.]+(?=,|$)' "$wer_file")
    if [ -z "$wer_score" ]; then
        echo "❌ Error: No valid WER value found in $wer_file"
        exit 1
    fi
    {
        echo {\'metricResult\': {\'CER/WER\': $wer_score, \'Speaker Similarity\': $spk_simi_score}}
        echo "CER/WER: $wer_score"
        echo "Speaker Similarity: $spk_simi_score"
    } | tee "${decode_dir}/${name_without_extension}/eval.log"
done


