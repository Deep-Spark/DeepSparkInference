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

from dltest.cli.assert_cli import AssertCLI
from dltest.cli.log_comparator_cli import LogComparatorCLI
from dltest.cli.model_validator_cli import ModelValidatorCLI
from dltest.cli.fetch_log_cli import FetchLog
from dltest.cli.check_cli import CheckCli


#log_comparator_cli = LogComparatorCLI()
#model_validator_cli = ModelValidatorCLI()
fetch_log_cli = FetchLog()
#assert_cli = AssertCLI()
#check_cli = CheckCli()


def make_execute_path():
    preffix = "dltest.cli.entry_points"
    clis = []
    for cli_var in globals():
        if cli_var.endswith('_cli'):
            cmd_name = globals()[cli_var].command_name()
            clis.append(f"ixdltest-{cmd_name}={preffix}:{cli_var}")

    return clis


