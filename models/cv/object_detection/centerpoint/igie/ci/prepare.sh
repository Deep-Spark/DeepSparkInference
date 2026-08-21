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

if [[ ! -d CenterPoint/.git ]]; then
  git clone https://github.com/tianweiy/CenterPoint.git
fi
rsync -a adapt/ CenterPoint/

cd CenterPoint
bash apply_compat.sh
pip3 install -r requirements.txt
bash setup.sh

if [[ -n "${DATASETS_DIR:-}" ]]; then
  mkdir -p data
  ln -sfn "${DATASETS_DIR}" data/nuScenes
fi
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  ln -sfn "${CHECKPOINT_PATH}" ./latest.pth
fi

if [[ -n "${TVM_HOME:-}" && -f "${TVM_HOME}/build/libtvm.so" && -f latest.pth && -d data/nuScenes ]]; then
  ./run_igie.sh build
fi
