# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

from os.path import join, dirname, exists
import tensorrt as trt
import ctypes

def load_ixrt_plugin(logger=trt.Logger(trt.Logger.WARNING), namespace="", dynamic_path=""):
    if not dynamic_path:
        dynamic_path = join(dirname(trt.__file__), "lib", "libixrt_plugin.so")
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, namespace)
    print(f"Loaded plugin from {dynamic_path}")