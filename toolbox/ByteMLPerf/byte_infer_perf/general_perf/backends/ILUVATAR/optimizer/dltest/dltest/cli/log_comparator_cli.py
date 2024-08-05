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
from pprint import pprint

from dltest.cli.log_parser_cli import LogParserCLI
from dltest.log_comparator import compare_logs_with_paths, DEFAULT_NEAREST_MATCH_CHARS


class LogComparatorCLI(LogParserCLI):

    def command_name(self):
        return "compare"

    def predefine_args(self):
        super(LogComparatorCLI, self).predefine_args()
        self.parser.add_argument('--log1', type=str, help="First log")
        self.parser.add_argument('--log2', type=str, help="Second log")
        self.parser.add_argument('--threshold', type=float, default=0.0001, help="Threshold")
        self.parser.add_argument('--only_last', type=int, default=1, help='Whether use the last result to compare')
        self.parser.add_argument('--saved', type=str, default=None, help='Save to path')
        self.parser.add_argument('--print_result', action="store_true", default=False, help='Whether print result')
        self.parser.add_argument('--allow_greater_than', action="store_true", default=False, help='Allow log1 greater than log2')

    def parse_args(self, *args, **kwargs):
        args = super(LogComparatorCLI, self).parse_args(*args, **kwargs)
        args.only_last = args.only_last >= 1

        return args

    def run(self):
        args = self.parse_args()
        satisfied, results = compare_logs_with_paths(
            log1=args.log1, log2=args.log2,
            threshold=args.threshold,
            patterns=args.patterns, pattern_names=args.pattern_names,
            use_re=args.use_re, nearest_distance=args.nearest_distance,
            start_line_pattern_flag=args.start_flag,
            end_line_pattern_flag=args.end_flag,
            only_last=args.only_last,
            split_pattern=args.split_pattern,
            split_sep=args.split_sep,
            split_idx=args.split_idx,
            allow_greater_than=True
        )

        if args.print_result:
            pprint(results)

        if satisfied:
            print("SUCCESS")
        else:
            print("FAIL")

        if args.saved is not None:
            with open(args.saved, 'w') as f:
                json.dump(results, f)




