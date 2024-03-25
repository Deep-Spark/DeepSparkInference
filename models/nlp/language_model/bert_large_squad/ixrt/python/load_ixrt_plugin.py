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

from os.path import join, dirname, exists, abspath
import tensorrt as trt
import ctypes
import os
import subprocess

def is_nvidia_platform():
    try:
        # 尝试运行 nvidia-smi
        subprocess.check_output(['nvidia-smi'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def load_ixrt_plugin(logger=trt.Logger(trt.Logger.WARNING), namespace="", dynamic_path=""):
    if not dynamic_path:
        if is_nvidia_platform():
            dynamic_path = join(dirname(abspath(__file__)), "..", "build", "libixrt_plugin.so") 
        else:
            dynamic_path = join(dirname(trt.__file__), "lib", "libixrt_plugin.so")
        
    if not exists(dynamic_path):
        raise FileNotFoundError(
            f"The ixrt_plugin lib {dynamic_path} is not existed, please provided effective plugin path!")
    handle = ctypes.CDLL(dynamic_path, mode=ctypes.RTLD_GLOBAL)
    handle.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initLibNvInferPlugins.restype = ctypes.c_bool
    handle.initLibNvInferPlugins(None, namespace.encode('utf-8'))
    print(f"Loaded plugin from {dynamic_path}")