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

import json
from typing import Mapping

from dltest.log_parser import LogParser, DEFAULT_NEAREST_MATCH_CHARS
from dltest.utils.base_cli import BaseCLI


class LogParserCLI(BaseCLI):

    def predefine_args(self):
        self.parser.add_argument('-p', '--patterns', nargs="*", type=str, default=None, help='Fetched patterns')
        self.parser.add_argument('-pn', '--pattern_names', nargs="*", type=str, default=None, help='The name of pattern')
        self.parser.add_argument('--use_re', action="store_true", default=False, help='Whether use regular expression')
        self.parser.add_argument('-d', '--nearest_distance', type=int, default=DEFAULT_NEAREST_MATCH_CHARS, help='The nearest distance of matched pattern')
        self.parser.add_argument('--start_flag', type=str, default=None, help='The flag of start to record log')
        self.parser.add_argument('--end_flag', type=str, default=None, help='The flag of stop to record log')
        self.parser.add_argument('--split_pattern', type=str, default=None, help='The pattern is used to match line')
        self.parser.add_argument('--split_sep', nargs="*", type=str, default=None, help='The seperator is used to split line')
        self.parser.add_argument('--split_idx', nargs="*", type=int, default=None, help='The index of split line')

    def parse_args(self, *args, **kwargs):
        args = super(LogParserCLI, self).parse_args(*args, **kwargs)

        return args

