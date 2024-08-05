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
import os.path as ospath
from pathlib import Path
import tempfile


class TemporaryFile:

    def __init__(self, with_open=False, mode='r'):
        self.name = None
        self.with_open = with_open
        self.mode = mode

        self.file = None

    def create(self):
        self.name = tempfile.mktemp()
        file_path = Path(self.name)
        file_path.touch()

    def delete(self):
        if self.name is not None and ospath.exists(self.name):
            os.unlink(self.name)

    def read(self):
        self._check_file_status()
        return self.file.read()

    def readlines(self):
        self._check_file_status()
        return self.file.readlines()

    def _check_file_status(self):
        if self.file is None:
            raise RuntimeError("File is closed, please reopen it.")

    def __enter__(self):
        self.create()
        if self.with_open:
            self.file = open(self.name, mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_open:
            self.file.close()
        self.delete()








