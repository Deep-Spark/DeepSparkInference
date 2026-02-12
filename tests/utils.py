# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import os
import shutil
import subprocess
import sys

def is_debug():
    is_debug_flag = os.environ.get("IS_DEBUG")
    if is_debug_flag and is_debug_flag.lower()=="true":
        return True
    else:
        return False

def ensure_numactl_installed():

    G_BIND_CMD = os.environ.get("BIND_CMD", "")

    if "numactl" not in G_BIND_CMD:
        return  

    if shutil.which("numactl") is not None:
        return  

    install_commands = [
        ["apt-get", "update"], ["apt-get", "install", "-y", "numactl"],
        ["yum", "install", "-y", "numactl"],
        ["dnf", "install", "-y", "numactl"]
    ]

    for i, cmd_list in enumerate(install_commands):
        try:
            if cmd_list[0] == "apt-get" and i == 0:
                subprocess.run(["apt-get", "update"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                continue
            elif cmd_list[0] == "apt-get" and i == 1:
                pass

            subprocess.run(cmd_list, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("install numactl completed.")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue  

    print("failed to auto install numactl.", file=sys.stderr)
    sys.exit(1)
