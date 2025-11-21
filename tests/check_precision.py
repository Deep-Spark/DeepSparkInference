# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import argparse
from transformers import AutoConfig
import torch

def get_model_precision(model_name_or_path):
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        dtype_str = getattr(config, "torch_dtype", None)
        if dtype_str:
            if dtype_str == torch.bfloat16:
                return "BF16"
            elif dtype_str == torch.float16:
                return "FP16"
            elif dtype_str == torch.float32:
                return "FP32"
            else:
                return f"OTHER ({dtype_str})"
    except Exception as e:
        print(f"[Warning] Could not load config for '{model_name_or_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description='Determine the precision of a Hugging Face model.')
    parser.add_argument('model_name', type=str, help='The name or path of the Hugging Face model.')
    
    args = parser.parse_args()
    
    precision = get_model_precision(args.model_name)
    print(f"The precision of the model '{args.model_name}' is likely: {precision}")

if __name__ == "__main__":
    main()