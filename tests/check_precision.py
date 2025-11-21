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
from transformers import AutoModel
import torch

def get_model_precision(model_name_or_path):
    """
    Attempts to determine the precision of a model based on its name and loaded properties.
    """
    # 1. Analyze model name for clues
    model_name_lower = model_name_or_path.lower()
    
    if 'int8' in model_name_lower:
        return 'INT8'
    if 'int4' in model_name_lower:
        return 'INT4'
    if 'fp16' in model_name_lower or 'float16' in model_name_lower:
        return 'FP16'
    if 'fp32' in model_name_lower or 'float32' in model_name_lower:
        return 'FP32'
    # Common quantization formats
    if 'gguf' in model_name_lower:
        # GGUF files often specify quantization level in their name, e.g., q4_k_m
        # If not specified, assume a common default or lower precision
        # For simplicity here, we'll return a generic 'QUANTIZED' or check further if needed
        # Let's assume if it's gguf, it's quantized, but default to checking dtype if possible
        pass # Continue to check dtype below

    # 2. Load model config and check dtype (requires loading, might be slow for large models)
    try:
        # Load the model (this can be heavy)
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            # torch_dtype=torch.float16, # Avoid forcing dtype here initially
            # low_cpu_mem_usage=True, # Optional, to reduce CPU memory usage during load
        )
        model_dtype = model.dtype
        if model_dtype == torch.int8:
            return 'INT8'
        elif model_dtype == torch.float16:
            return 'FP16'
        elif model_dtype == torch.float32:
            return 'FP32'
        elif model_dtype == torch.bfloat16:
            return 'BF16'
        else:
            # For other types like INT4 (if represented differently) or custom dtypes
            return f'OTHER ({model_dtype})'
    except Exception as e:
        print(f"Warning: Could not load model '{model_name_or_path}' to check dtype: {e}")
        print("Attempting to infer from name again or defaulting...")
        # If loading fails, return a default guess based on common conventions
        # Most standard models without explicit quantization in name are FP16
        return 'FP16 (Inferred from common default, loading failed)'

def main():
    parser = argparse.ArgumentParser(description='Determine the precision of a Hugging Face model.')
    parser.add_argument('model_name', type=str, help='The name or path of the Hugging Face model.')
    
    args = parser.parse_args()
    
    precision = get_model_precision(args.model_name)
    print(f"The precision of the model '{args.model_name}' is likely: {precision}")

if __name__ == "__main__":
    main()