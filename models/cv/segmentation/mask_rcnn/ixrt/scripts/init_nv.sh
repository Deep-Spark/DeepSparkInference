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

# build plugin
cd ../plugins
rm -rf build
mkdir build
cd build
cmake .. -DUSE_TRT=1 -DNV_TRT_PATH=/usr/local/TensorRT
make -j8

## install packages
bash prepare_system_env.sh

# pip whl
pip3 install opencv-python==4.6.0.66
pip3 install pycuda
pip3 install pycocotools
pip3 install tqdm
pip3 install torch
pip3 install torchvision

# build engine
cd ../../python
rm -rf ./maskrcnn.engine
python3 maskrcnn.py build_engine --wts_file ./maskrcnn.wts --engine_file ./maskrcnn.engine

