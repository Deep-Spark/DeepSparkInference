#!/bin/bash
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# set -euox pipefail

data_dir=$1
tool_dir=$2

wav_dir=${data_dir}/wav
aishell_text=${data_dir}/transcript/aishell_transcript_v0.8.txt

# data directory check
if [ ! -d $wav_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: wav directory and aishell text not found!"
  exit 1;
fi

# find test wav file
local_dir=${data_dir}/local
mkdir -p $local_dir
find $wav_dir -iname "*.wav" > $local_dir/wav.flist || exit 1;

# Transcriptions preparation
sed -e 's/\.wav//' $local_dir/wav.flist | awk -F '/' '{print $NF}' > $local_dir/utt.list
paste -d' ' $local_dir/utt.list $local_dir/wav.flist > $local_dir/wav.scp_all
${tool_dir}/filter_scp.pl -f 1 $local_dir/utt.list $aishell_text > $local_dir/transcripts.txt
awk '{print $1}' $local_dir/transcripts.txt > $local_dir/utt.list
${tool_dir}/filter_scp.pl -f 1 $local_dir/utt.list $local_dir/wav.scp_all | sort -u > $local_dir/wav.scp
sort -u $local_dir/transcripts.txt > $local_dir/text
echo "Preparing transcriptions succeeded!"

test_dir=${data_dir}/test
mkdir -p ${test_dir}
for f in wav.scp text; do
  cp $local_dir/$f ${test_dir}/$f || exit 1;
done
rm -r ${data_dir}/local

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

# remove the space between the text labels for Mandarin dataset
cp $test_dir/text $test_dir/text.org
paste -d " " <(cut -f 1 -d" " ${test_dir}/text.org) \
  <(cut -f 2- -d" " ${test_dir}/text.org | tr -d " ") \
  > ${test_dir}/text
rm ${test_dir}/text.org

# Prepare required format
if [ $data_type == "shard" ]; then
  ${tool_dir}/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
    --num_threads 16 $test_dir/wav.scp $test_dir/text \
    $(realpath $test_dir/shards) $test_dir/data.list
else
  ${tool_dir}/make_raw_list.py $test_dir/wav.scp $test_dir/text \
    $test_dir/data.list
fi

echo "AISHELL data preparation succeeded!"