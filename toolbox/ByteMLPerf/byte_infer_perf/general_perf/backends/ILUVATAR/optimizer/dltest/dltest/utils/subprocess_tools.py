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

import subprocess
from typing import Callable, Union, List

from dltest.utils.real_tempfile import TemporaryFile
from dltest.utils import misc


def get_output_with_pipe(command, shell=None, callback: Callable[[list], None]=None, *args, **kwargs):
    if shell is None:
        shell = True

    if shell and not isinstance(command, str):
        command = " ".join(command)

    stream = subprocess.Popen(
        command, shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        *args, **kwargs
    )
    outputs = []
    while 1:
        exit_code = stream.poll()
        if exit_code is None:
            if stream.stdout.readable():
                outputs.append(stream.stdout.readline().decode("utf8").rstrip())
                if callback is not None:
                    callback(outputs[-1:])
                print(outputs[-1])
        else:
            if stream.stdout.readable():
                lines = stream.stdout.readlines()
                lines = [line.decode("utf8".rstrip()) for line in lines]
                outputs.extend(lines)
                if callback is not None:
                    callback(outputs[-1:])
                print('\n'.join(lines))
            break

    return outputs


def get_output_with_tempfile(command, *args, **kwargs):
    if not isinstance(command, (list, tuple)):
        command = [command]
    stdout = None
    with TemporaryFile(with_open=True) as file:
        command.extend(['|', 'tee', file.name])
        command = " ".join(command)

        res = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, shell=True, *args, **kwargs)
        output = file.readlines()

    return output

def execute_shell(command, *args, **kwargs):
    if "env" not in kwargs:
        kwargs["env"] = misc.create_subproc_env()

    if not isinstance(command, (list, tuple)):
        command = [command]

    command = " ".join(command)
    res = subprocess.run(command,
                         shell=True, *args, **kwargs)
    return res

def get_output(command: List, capture_output_method: str = 'tempfile', *args, **kwargs):
    if "env" not in kwargs:
        kwargs["env"] = misc.create_subproc_env()

    if capture_output_method == "tempfile":
        return get_output_with_tempfile(command, *args, **kwargs)
    return get_output_with_pipe(command, *args, **kwargs)