#!/bin/bash

pip3 install transformers datasets h5py==3.1.0 tqdm argparse -U

DIR_NAME='test_models'

if [[ ! -d ${DIR_NAME} ]]; then
    mkdir -p ${DIR_NAME}
fi

if [[ ! -f "${DIR_NAME}/vocab.txt" ]]; then
    wget 'http://10.113.3.3/data/Model/bert_squad/vocab.txt' -P ${DIR_NAME}
fi

if [[ ! -f "${DIR_NAME}/train.json" ]]; then
    wget 'http://10.113.3.3/data/Model/bert_squad/train.json' -P ${DIR_NAME}
fi

if [[ ! -f "${DIR_NAME}/dev.json" ]]; then
    wget 'http://10.113.3.3/data/Model/bert_squad/dev.json' -P ${DIR_NAME}
fi


if [[ ! -f "${DIR_NAME}/tokenizer_config.json" ]]; then
    wget 'http://10.113.3.3/data/Model/bert_squad/tokenizer_config.json' -P ${DIR_NAME}
fi

if [[ ! -f "${DIR_NAME}/config.json" ]]; then
    wget 'http://10.113.3.3/data/Model/bert_squad/config.json' -P ${DIR_NAME}
fi

# model_name="base"

# if [[ ${model_name} == 'base' ]]; then
#     if [[ ! -f "${DIR_NAME}/bert_base_quant.hdf5" ]]; then
#         wget 'http://10.113.3.3/data/Model/bert_squad/bert_base_quant.hdf5' -P ${DIR_NAME}
#         wget 'http://10.113.3.3/data/Model/bert_squad/bert_base_quant.hdf5.md5' -P ${DIR_NAME}
#     fi
# fi

# model_name="large"

# if [[ ${model_name} == 'large' ]]; then
#     if [[ ! -f "${DIR_NAME}/bert_large_quant.hdf5" ]]; then
#         wget 'http://10.113.3.3/data/Model/bert_squad/bert_large_quant.hdf5' -P ${DIR_NAME}
#         wget 'http://10.113.3.3/data/Model/bert_squad/bert_large_quant.hdf5.md5' -P ${DIR_NAME}
#     fi
# fi