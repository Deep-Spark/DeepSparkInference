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

import argparse

import numpy as np
import torch


def gen_data(batch_size, output):
    a = torch.randint(0, 50265, (batch_size, 384))
    a = a.numpy().astype(np.int64)
    a.tofile(output+"input_ids.bin")

    a = np.ones((batch_size, 384), dtype=np.int64)
    a.tofile(output+"input_mask.bin")

    a = np.zeros((batch_size, 384), dtype=np.int64)
    a.tofile(output+"token_type_ids.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data for RoBERTa model.")
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for data generation"
    )
    parser.add_argument("--output_path", default="")

    args = parser.parse_args()

    gen_data(args.batch_size, args.output_path)