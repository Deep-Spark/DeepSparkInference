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
import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# cpp_files = glob.glob(os.path.join(CUR_DIR,"*.cpp"))
# cu_files = glob.glob(os.path.join(CUR_DIR,'*.cu'))
# source_files = cpp_files + cu_files
# print("source files:")
# for i in source_files:
#     print(i)
source_files = [
    os.path.join(CUR_DIR,'test.cpp'),
    os.path.join(CUR_DIR,'kernel.cu'),
]

for i in source_files:
    assert os.path.isfile(i)
    print(i)

setup(
    name="test",
    ext_modules=[
        CUDAExtension(
            name="sp_opt",
            libraries=["cuinfer"],
            sources=source_files)
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
