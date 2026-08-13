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
set -x

meta_lst="$1"
output_dir="$2"
lang="$3"
num_job="$4"

if ! [[ "$num_job" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: num_job must be a positive integer, got: $num_job" >&2
    exit 1
fi

wav_wav_text="$output_dir/wav_res_ref_text"
score_file="$output_dir/wav_res_ref_text.wer"

workdir=$(cd "$(dirname "$0")"; cd ../; pwd)

python3 get_wav_res_ref_text.py "$meta_lst" "$output_dir/wavs" "$wav_wav_text"
# python3 prepare_ckpt.py

timestamp=$(date +%s)
thread_dir="${output_dir}/tmp/thread_metas_${timestamp}"
out_dir="${thread_dir}/results"

mkdir -p "$out_dir"

num=$(wc -l < "$wav_wav_text")
if (( num == 0 )); then
    echo "ERROR: No generated WAV files were found in $output_dir/wavs; WER cannot be calculated." >&2
    exit 1
fi

num_per_thread=$((num / num_job + 1))
split -l "$num_per_thread" --additional-suffix=.lst -d \
    "$wav_wav_text" "$thread_dir/thread-"

pids=()
rank=0
for thread_file in "$thread_dir"/thread-*.lst; do
    thread_name=$(basename "${thread_file%.lst}")
    sub_score_file="$out_dir/${thread_name}.wer.out"
    CUDA_VISIBLE_DEVICES=$((rank % num_job)) \
        python3 run_wer.py "$thread_file" "$sub_score_file" "$lang" &
    pids+=("$!")
    rank=$((rank + 1))
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

# rm $wav_wav_text
# rm -f $out_dir/merge.out

cat "$out_dir"/thread-*.wer.out > "$out_dir/merge.out"
python3 average_wer.py "$out_dir/merge.out" "$score_file"