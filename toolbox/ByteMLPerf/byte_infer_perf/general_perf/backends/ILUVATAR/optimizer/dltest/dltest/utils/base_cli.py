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

from argparse import ArgumentParser
from abc import abstractmethod


class BaseCLI:

    def __init__(self, parser=None, *args, **kwargs):
        if parser is None:
            self.parser = ArgumentParser(description=self.description ,*args, **kwargs)

    def __call__(self):
        self.run()

    @property
    def description(self):
        return None

    @abstractmethod
    def command_name(self):
        pass

    def predefine_args(self):
        pass

    def parse_args(self, *args, **kwargs):
        self.predefine_args()
        return self.parser.parse_args(*args, **kwargs)

    @abstractmethod
    def run(self):
        pass



