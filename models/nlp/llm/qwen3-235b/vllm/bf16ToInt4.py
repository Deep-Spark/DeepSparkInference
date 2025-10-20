#!/bin/bash
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
#
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file
import ixformer.inference.functions as ixfop
import shutil
import json

def weight_quant(v: torch.Tensor):
    assert v.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (v.shape[0], 1)
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)

def weight_quant_moe(v: torch.Tensor,  block_size: int = 128, group_size=-1, format="TN", symmetric=True, version=2):
    assert v.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (v.shape[0], 1)
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    quantized = quantized.to(torch.int8)
    i4_weights, i8scales, i8zeros = ixfop.quant_repack_int4(quantized.to(torch.int8).unsqueeze_(0), group_size, version, format, not symmetric)
    return i4_weights.squeeze(0), scale.to(torch.float32), i8scales, i8zeros


def main(fp8_path, int8_path, group_size, format, symmetric, version, split_count):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    int8_path (str): The path to the directory where the converted int8 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []


    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    
    new_weight_map = {}
    all_safetensor = safetensor_files
    all_files = list(glob(os.path.join(fp8_path, "*")))
    if split_count is None:
        
        safetensor_files = safetensor_files
    elif split_count == 1:
        safetensor_files = safetensor_files[:-2]
    else:
        safetensor_files = safetensor_files[-2:]
    

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if "experts" in weight_name:
                i4_weights, scale, i8scales, i8zeros = weight_quant_moe(weight, 128, group_size, format, symmetric, version)
                if version == 2:
                    scale = scale.contiguous().view(1, -1)
                else:
                    assert scale.is_contiguous()
                new_state_dict[weight_name] = i4_weights
                sacle_name = weight_name.replace("weight","weight_scale")
                new_state_dict[sacle_name] = scale
                
                new_weight_map[weight_name] = file_name
                new_weight_map[sacle_name] = file_name
                
                
                if i8scales is not None:
                    i8scales = i8scales.squeeze_(0)
                    assert i8scales.dim() == 2
                    
                    i8scales_name = weight_name.replace("weight","i8_weight_scale")
                    new_state_dict[i8scales_name] = i8scales
                    new_weight_map[i8scales_name] = file_name
                    
                    
                if i8zeros is not None:
                    i8zeros = i8zeros.squeeze_(0)
                    assert i8zeros.dim() == 2
                    i8zeros_name = weight_name.replace("weight","i8_weight_zero")
                    new_state_dict[i8zeros_name] = i8zeros
                    new_weight_map[i8zeros_name] = file_name

            elif "proj" in weight_name:
                int8_v, scale = weight_quant(weight)
                new_state_dict[weight_name] = int8_v
                new_scale_name = weight_name + "_scale"
                new_state_dict[new_scale_name] = scale
                
                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
                      
        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 1:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
            
    other_files = list(set(all_files) - set(all_safetensor))
    for other_file in other_files:
        if os.path.isfile(other_file):
            name = other_file.rsplit("/", 1)[1]
            shutil.copy(os.path.join(other_file),
                        os.path.join(int8_path, name))
    
    compression_config = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None, 
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int"
                },
                "output_activations": None,
                "targets": [
                    "Linear"
                ],
                "weights": {
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None if group_size==-1 else group_size,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel" if group_size == -1 else "group",
                    "symmetric": bool(symmetric),
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": 1.0,
        "ignore": [
            "lm_head"
        ],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen"
    }
    
    with open(os.path.join(int8_path, "config.json"), encoding="utf-8") as file:
        configs:dict = json.loads(file.read())
        # configs.pop("quantization_config")
        configs["compression_config"] = compression_config
    with open(os.path.join(int8_path, "config.json"), encoding="utf-8", mode="w") as f:
        json.dump(configs, f)
        
        
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    new_model_index_file = os.path.join(int8_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {new_model_index_file}")    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=-1)
    parser.add_argument("--format", type=str, default="TN")
    parser.add_argument("--symmetric", type=bool, default=True)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--split-count", type=int, default=None)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_int8_hf_path, args.group_size, args.format, args.symmetric, args.version, args.split_count)