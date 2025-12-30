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

inference_dir=$1
data_dir=$2
output_dir=$3

OS_ID=$(awk -F= '/^ID=/{print $2}' /etc/os-release | tr -d '"')
if [[ "$OS_ID" == "ubuntu" ]]; then
    sudo apt-get install -y sox libsox-dev libmagic1 libmagic-dev libgl1 libglib2.0-0
    sudo apt-get update && sudo apt-get install -y ffmpeg
elif [[ "$OS_ID" == "centos" ]]; then
    sudo yum install -y sox sox-devel file-devel mesa-libGL
    sudo yum install -y epel-release
    sudo yum install -y https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
    sudo yum install -y https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm
    sudo yum install -y ffmpeg ffmpeg-devel
    ffmpeg -version
fi

REPO_DIR="CosyVoice"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning CosyVoice repository for the first time..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd "$REPO_DIR" || exit 1
    git submodule update --init --recursive
    echo "Repository cloned and submodules initialized."
else
    echo "CosyVoice repository already exists. Skipping git clone."
    cd "$REPO_DIR"
    # git pull && git submodule update --init --recursive
fi

model_name="CosyVoice2-0.5B"
target_dir="pretrained_models/${model_name}"
if [ -d "${target_dir}" ] && [ -n "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
    echo "✅ Model already exists at: ${target_dir}"
    echo "   Skipping download."
else
    echo "Preparing to download model: iic/${model_name} to ${target_dir}"
    mkdir -p "${target_dir}"
    python3 -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='${target_dir}')
" || {
        echo "❌ ERROR: Model download failed (non-zero exit code from Python)."
        exit 1
    }
    if [ ! -d "${target_dir}" ] || [ -z "$(ls -A "${target_dir}" 2>/dev/null)" ]; then
        echo "❌ ERROR: Downloaded directory is empty or does not exist: ${target_dir}"
        exit 1
    fi
    echo "✅ Model downloaded successfully to: ${target_dir}"
fi

mkdir -p "${output_dir}"
cp -f ../scripts/get_infer_wavs.py .
python3 get_infer_wavs.py \
--inference_dir $inference_dir \
--input_text $inference_dir/$data_dir/text \
--prompt_text $inference_dir/$data_dir/prompt_text \
--prompt_wav_scp $inference_dir/$data_dir/prompt_wav.scp \
--output_dir $output_dir \
--fp16

cd ..
