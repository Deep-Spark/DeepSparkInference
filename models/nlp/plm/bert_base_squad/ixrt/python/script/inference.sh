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

PASSAGE='TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, 
speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations 
for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components 
to take advantage of powerful TensorRT optimizations for your apps.'
QUESTION="What is TensorRT?"

USE_FP16=True

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --int8) USE_FP16=False;;
    esac
done

if [ "$USE_FP16" = "True" ]; then
    echo 'USE_FP16=True'
    python3 inference.py -e ./data/bert_base_384.engine \
                        -s 384 \
                        -p $PASSAGE \
                        -q $QUESTION \
                        -v ./data/bert_base_uncased_squad/vocab.txt 
else
    echo 'USE_INT8=True'
    python3 inference.py -e ./data/bert_base_384_int8.engine \
                        -s 384 \
                        -p $PASSAGE \
                        -q $QUESTION \
                        -v ./data/bert_base_uncased_squad/vocab.txt 
fi

