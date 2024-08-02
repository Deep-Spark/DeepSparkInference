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

VERSION='v1.1'

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_squad.sh [v2_0|v1_1]"
            exit 0
            ;;
        v2_0) VERSION='v2.0'
            ;;
        v1_1) VERSION='v1.1'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Download the SQuAD training and dev datasets
echo "Step 1: Downloading SQuAD-${VERSION} training and dev datasets to ./data/squad"
if [ ! -d "./data" ]; then
    mkdir -p data
else
    echo 'data directory existed'
fi

pushd data
if [ ! -d "./squad" ]; then
    mkdir -p squad
    pushd squad
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-${VERSION}.json
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-${VERSION}.json
    popd
else 
    echo 'squad directory existed'
fi

echo "Step 2: Downloading model file and config to ./data/bert_base_uncased_squad"

if [ ! -d "./bert_base_uncased_squad" ]; then
    wget https://drive.google.com/file/d/1_q7SaiZjwysJ3jWAIQT2Ne-duFdgWivR/view?usp=drive_link
    unzip bert_base_uncased_squad.zip -d ./
    rm -f bert_base_uncased_squad.zip
else 
    echo 'bert_base_uncased_squad directory existed'
fi
popd
