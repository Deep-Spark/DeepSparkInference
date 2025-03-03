#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

ln -s /root/data/checkpoints/maskrcnn.wts ./python/
ln -s /root/data/datasets/coco ./coco
# install ixrt run
bash /root/data/3rd_party/ixrt-0.10.0+corex.4.2.0.20250115-linux_x86_64.run

if [ "$1" = "nvidia" ]; then
    cd scripts && bash init_nv.sh
else
    cd scripts && bash init.sh
fi