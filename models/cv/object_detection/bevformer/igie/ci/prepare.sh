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

if [[ ! -d nuCarla/.git ]]; then
  git clone https://github.com/michigan-traffic-lab/nuCarla.git
fi
rsync -a adapt/ nuCarla/BEVFormer/

cd nuCarla/BEVFormer
[[ -d mmcv ]] && mv mmcv mmcv-src || true
pip3 install -r requirements.txt || true
python3 setup.py develop --user 2>/dev/null || true

mkdir -p ckpts
if [[ ! -f ckpts/bevformer-base.pth ]]; then
  wget -q https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevformer-base.pth \
    -O ckpts/bevformer-base.pth || true
fi

if [[ -n "${DATASETS_DIR:-}" ]]; then
  mkdir -p data
  ln -sfn "${DATASETS_DIR}" data/nuscenes
fi

if [[ -n "${TVM_HOME:-}" && -f "${TVM_HOME}/build/libtvm.so" && -f ckpts/bevformer-base.pth && -d data/nuscenes ]]; then
  export PYTHONPATH="${PWD}:${TVM_HOME}/python:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="${TVM_HOME}/build:/opt/sw_home/local/corex/lib64:${LD_LIBRARY_PATH:-}"
  ./run_igie.sh build
fi
