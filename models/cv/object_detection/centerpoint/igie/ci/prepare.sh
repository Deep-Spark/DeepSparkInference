#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f /opt/sw_home/enable ]]; then
  # shellcheck disable=SC1091
  source /opt/sw_home/enable
fi

cp -r /mnt/deepspark/data/3rd_party/CenterPoint ./
cp -r adapt/* CenterPoint/

cd CenterPoint
bash apply_compat.sh
pip3 install -r requirements.txt
bash setup.sh

mkdir -p data
ln -s /mnt/deepspark/data/datasets/nuscenes data/
ln -s /mnt/deepspark/data/checkpoints/latest.pth ./

pip3 install spconv
./run_igie.sh build

