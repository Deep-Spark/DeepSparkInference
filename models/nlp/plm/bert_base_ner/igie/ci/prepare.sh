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

pip3 uninstall numpy
pip3 install -r requirements.txt

# Get pytorch weights
# python3 get_weights.py

# Do QAT for INT8 test, will take a long time  
cd Int8QAT
python3 run_qat.py --model_dir ../test/ --datasets_dir ${DATASETS_DIR}
python3 export_hdf5.py --model quant_base/pytorch_model.bin