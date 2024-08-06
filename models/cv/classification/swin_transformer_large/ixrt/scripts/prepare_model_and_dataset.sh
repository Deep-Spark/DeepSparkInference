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

# #!/bin/bash
echo "******************* Downloading Model....  *******************"

mkdir -p general_perf/model_zoo/regular
mkdir -p general_perf/model_zoo/popular
mkdir -p general_perf/model_zoo/sota
mkdir -p general_perf/download
mkdir -p datasets/open_imagenet/

wget -O general_perf/download/open-swin-large.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open-swin-large.tar
tar xf general_perf/download/open-swin-large.tar -C general_perf/model_zoo/popular/


# # Download Datasets
wget -O general_perf/download/open_imagenet.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_imagenet.tar
tar xf general_perf/download/open_imagenet.tar -C datasets/


echo "Extract Done."
