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

# Backbone factory aligned with InsightFace arcface_torch:
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/__init__.py

from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    if name == "r18":
        return iresnet18(False, **kwargs)
    if name == "r34":
        return iresnet34(False, **kwargs)
    if name == "r50":
        return iresnet50(False, **kwargs)
    if name == "r100":
        return iresnet100(False, **kwargs)
    if name == "r200":
        return iresnet200(False, **kwargs)
    raise ValueError("unsupported network: {}".format(name))
