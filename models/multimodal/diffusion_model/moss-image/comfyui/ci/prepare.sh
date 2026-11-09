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
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
git checkout v0.44.0

# --no-deps keeps the image's own CoreX PyTorch in place.
pip3 install --no-deps torchsde trampoline alembic SQLAlchemy Mako greenlet av simpleeval

# comfy-aimdo and comfy-kitchen are imported at startup. Take only the pure Python
# wheels: the binary ones carry NVIDIA CUDA kernels. Both fall back to plain PyTorch.
pip3 download --no-deps --platform any --only-binary=:all: -d wheels comfy-aimdo==0.6.0 comfy-kitchen==0.2.40
pip3 install --no-deps wheels/*.whl

# An empty front-end root avoids needing comfyui-frontend-package.
mkdir -p web
