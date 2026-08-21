#!/bin/bash

# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IGIE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${REPO_DIR:-${IGIE_DIR}/nuCarla/BEVFormer}"

if [[ -z "${TVM_HOME:-}" ]]; then
  echo "error: export TVM_HOME=/path/to/igie before running this script"
  exit 1
fi

if [[ ! -x "${REPO_DIR}/run_igie.sh" ]]; then
  echo "error: ${REPO_DIR}/run_igie.sh not found."
  echo "  cd ${IGIE_DIR} && git clone https://github.com/michigan-traffic-lab/nuCarla.git && rsync -a adapt/ nuCarla/BEVFormer/"
  exit 1
fi

cd "${REPO_DIR}"
./run_igie.sh build
./run_igie.sh accuracy
