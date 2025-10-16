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
import shutil
import torch
from safetensors.torch import load_file, save_file

def weight_quant(v: torch.Tensor,  block_size: int = 128):
    
    #TODO Pdd 128 group?
    if v.dim() == 2:
        qmax = 127.0
        abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
        scale = abs_max / qmax  # [rows, 1]
        assert scale.shape == (v.shape[0], 1)
        quantized = torch.round(v / scale)
        quantized = torch.clamp(quantized, -qmax, qmax)
        return quantized.to(torch.int8), scale.to(torch.float32)
    elif v.dim() == 3:
        qmax = 127.0
        scales = torch.empty(v.shape[0], v.shape[1], 1).to(device=v.device, dtype=torch.float32)
        quantized = torch.empty_like(v, dtype=torch.int8)
        for i in range(v.shape[0]):
            abs_max = torch.abs(v[i]).max(dim=1, keepdim=True)[0]
            scales[i] = abs_max / qmax
            quantized[i] = torch.round(v[i] / scales[i])
            quantized[i] = torch.clamp(quantized[i], -qmax, qmax)
        return quantized, scales
        

def process_config_weight_map(fp8_path, int8_path):
    config_path = os.path.join(fp8_path, "config.json")
    config_save_path = os.path.join(int8_path, "config.json")
    
    with open(config_path, "r") as f_open:
        config = json.load(f_open)
        # del config["quantization_config"]
        config["compression_config"] = {"config_groups": {
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
                                        "group_size": None,
                                        "num_bits": 8,
                                        "observer": "minmax",
                                        "observer_kwargs": {},
                                        "strategy": "channel",
                                        "symmetric": True,
                                        "type": "int"
                                        }
                                    }
                                    },
                                    "format": "int-quantized",
                                    "global_compression_ratio": 1.2405352996226195,
                                    "ignore": [
                                    "lm_head"
                                    ],
                                    "kv_cache_scheme": None,
                                    "quant_method": "compressed-tensors",
                                    "quantization_status": "frozen"
                                }
        with open(config_save_path, "w") as f_save:
            json.dump(config, f_save, indent=4)
def main(fp8_path, int8_path):
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

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]
        
    files = os.listdir(fp8_path)
    for file in files:
        if not os.path.isdir(os.path.join(fp8_path, file)) and not file.endswith("safetensors") and file != "model.safetensors.index.json" and file != "config.json":
          file_path = os.path.join(fp8_path, file)
          save_file_path = os.path.join(int8_path, file)
          shutil.copy(file_path, save_file_path)
    # modify config.json
    process_config_weight_map(fp8_path, int8_path)
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    
    safetensor_files.sort()
    new_weight_map = {}
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if (not "vision_model" in weight_name and not "vit" in weight_name) and ("proj" in weight_name or "wq" in weight_name):       
                try:
                    int8_v, scale = weight_quant(weight)
                    new_state_dict[weight_name] = int8_v
                    new_scale_name = weight_name + "_scale"
                    new_state_dict[new_scale_name] = scale
                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
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
    # modify model.safetensors.index.json
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    new_model_index_file = os.path.join(int8_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {new_model_index_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    parser.add_argument("--split-count", type=int, default=None)
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_int8_hf_path)


#python3 bf162int8.py --input-bf16-hf-path /data/nlp/ckpt4_bf16/  --output-int8-hf-path /data/nlp/ckpt4_int8/     