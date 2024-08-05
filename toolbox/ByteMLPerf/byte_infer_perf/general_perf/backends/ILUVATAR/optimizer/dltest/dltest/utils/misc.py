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

import copy
import os


def get_full_path(fname):
    pwd = os.getcwd()
    if fname.startswith('/'):
        return fname
    return os.path.join(pwd, fname)


def is_main_proc(rank):
    return str(rank) in ["None", "-1", "0"]


def main_proc_print(*args, **kwargs):
    if "RANK" in os.environ:
        if is_main_proc(os.environ["RANK"]):
            print(*args, **kwargs)
            return

    if "LOCAL_RANK" in os.environ:
        if is_main_proc(os.environ["LOCAL_RANK"]):
            print(*args, **kwargs)
            return

    print(*args, **kwargs)


def create_subproc_env():
    env = copy.copy(os.environ)
    env["USE_DLTEST"] = "1"
    return env