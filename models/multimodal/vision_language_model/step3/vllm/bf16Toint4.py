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


def quant_repack_int4(x, group_size, version, format, isAsymQuant: bool = False):
    n_experts, n, k = x.shape
    if version == 1:
        assert not isAsymQuant

        if group_size == -1:
            max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
            scales = torch.round(max_x / 7)
            scales[scales < 1e-6] = 1
            out = torch.round(x / scales).clamp(-8, 7).to(torch.int8)
        else:
            x = x.view(n_experts, -1, group_size)
            max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
            scales = torch.round(max_x / 7)
            scales[scales < 1e-6] = 1
            out = torch.round(x / scales).clamp(-8, 7).to(torch.int8)

        out = out.view(n_experts, n, k)

        if format[0] == "N":
            out = out.transpose(-2, -1).contiguous()  # NT (num_experts, k , n)
            out = out.reshape(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.view(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(n_experts, k, n)

        ## rearange 32 token
        shape = out.shape
        out = out.view(shape[0], shape[1], shape[-1] // 32, 32)
        out_tmp = out.new_empty(shape[0], shape[1], shape[-1] // 32, 16)
        for i in range(16):
            sign_low_4bit = (out[:, :, :, i] < 0).to(torch.int8)
            low_4bit = sign_low_4bit * 8 + (out[:, :, :, i] & 0x07)
            high_4bit = out[:, :, :, i + 16] << 4
            out_tmp[:, :, :, i] = high_4bit + low_4bit
        out = out_tmp.view(shape[0], shape[1], shape[-1] // 2).contiguous()

        scales = (
            scales.view(n_experts, n, k // group_size).permute(0, 2, 1).contiguous()
            if group_size != -1
            else scales.view(n_experts, n)
        )

        return out, scales, None

    if version == 2:
        """
        For group_size == -1 (per-channel), the default scale factor is 18 since
        127 / 7 = 18, for quantization with clip, the scale can be set to 16, 17, etc.
        the alpha in ixinfer_gemm_helper need to be set to scale / 16.0, and the ixformer
        need to be rebuilt.
        """
        if group_size == -1:
            out = torch.round(x.cpu() / 18).clamp(-8, 7).to(torch.int8)
        else:
            x = x.view(n_experts, -1, group_size)
            if isAsymQuant:
                max_x, _ = torch.max(x, dim=-1, keepdim=True)
                min_x, _ = torch.min(x, dim=-1, keepdim=True)
                scales = ((max_x.to(torch.float32) - min_x.to(torch.float32)) / 15).to(
                    torch.int8
                )
                zeros = (-min_x / scales - 8).to(
                    torch.int8
                )  # weight use int4 not uint4, and zero use int8
                out = (x / scales + zeros).clamp(-8, 7).to(torch.int8)
            else:
                max_x, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
                scales = torch.round(max_x / 7)
                scales[scales < 1e-6] = 1
                scales = scales.to(torch.int8)
                #.cpu() avoid oom
                out = torch.round(x.cpu() / scales.cpu()).clamp(-8, 7).to(torch.int8)
                scales = scales.to(x.device)
                
            out = out.view(n_experts, n, k).contiguous()

        if format[0] == "N":
            out = out.transpose(-2, -1).contiguous()  # NT (num_experts, k , n)
            out = out.reshape(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.view(n_experts, k // 32, 2, 16, n // 32, 2, 16)
            out = out.permute(0, 1, 5, 3, 4, 2, 6).contiguous().view(n_experts, k, n)
            out = out.to(x.device)

        ## rearange 32 token
        shape = out.shape
        out = out.view(shape[0], shape[1], shape[-1] // 32, 32)
        out_tmp = out.new_empty(shape[0], shape[1], shape[-1] // 32, 16)
        for i in range(16):
            sign_low_4bit = (out[:, :, :, i] < 0).to(torch.int8)
            low_4bit = sign_low_4bit * 8 + (out[:, :, :, i] & 0x07)
            high_4bit = out[:, :, :, i + 16] << 4
            out_tmp[:, :, :, i] = high_4bit + low_4bit
        out = out_tmp.view(shape[0], shape[1], shape[-1] // 2).contiguous()

        if group_size == -1:
            return out, None, None

        scales = scales.to(torch.uint8)
        scales_4i8pack = scales.clone().to(torch.int32)
        for i in range(3):
            scales_4i8pack <<= 8
            scales_4i8pack |= scales
        scales_4i8pack = (
            scales_4i8pack.view(n_experts, n, k // group_size)
            .permute(0, 2, 1)
            .contiguous()
        )

        if not isAsymQuant:
            return out, scales_4i8pack, None

        zeros = zeros.to(torch.uint8)
        zeros_4i8pack = zeros.clone().to(torch.int32)
        for i in range(3):
            zeros_4i8pack <<= 8
            zeros_4i8pack |= zeros
        zeros_4i8pack = (
            zeros_4i8pack.view(n_experts, n, k // group_size)
            .permute(0, 2, 1)
            .contiguous()
        )

        return out, scales_4i8pack, zeros_4i8pack



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



def weight_dequant_moe(v: torch.Tensor,  block_size: int = 128, group_size=-1, format="TN", symmetric=True, version=2):
    if v.dim() == 2:
        qmax = 127.0
        abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
        scales = abs_max / qmax  # [rows, 1]
        assert scales.shape == (v.shape[0], 1)
        quantized = torch.round(v / scales)
        quantized = torch.clamp(quantized, -qmax, qmax)
        quantized = quantized.to(torch.int8)
    elif v.dim() == 3:
        qmax = 127.0
        scales = torch.empty(v.shape[0], v.shape[1], 1 ).to(device=v.device, dtype=torch.float32)
        quantized = torch.empty_like(v, dtype=torch.int8)
        for i in range(v.shape[0]):
            abs_max = torch.abs(v[i]).max(dim= 1, keepdim=True)[0]
            scales[i] = abs_max / qmax
            quantized[i] = torch.round(v[i] / scales[i])
            quantized[i] = torch.clamp(quantized[i], -qmax, qmax)    
        quantized = quantized.to(torch.int8)
        scales = scales.transpose(-2, -1).contiguous()  #  (48, 5120, 1) → (48, 1, 5120) #NN 
    assert quantized.dim() in (2, 3), f"Expected quantized to have 2 or 3 dimensions, but got {quantized.dim()}"        
    if  quantized.dim() == 2:
        quantized = quantized.unsqueeze(0)            
    i4_weights, i8scales, i8zeros = quant_repack_int4(quantized, group_size, version, format, not symmetric)
    
    return i4_weights.squeeze(0), scales.to(torch.float32), i8scales, i8zeros





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
            
            if (not "vision_model" in weight_name and not "vit" in weight_name and not "moe" in weight_name) and ("proj" in weight_name or "wq" in weight_name):       
                
                int8_v, scale = weight_quant(weight)
                new_state_dict[weight_name] = int8_v
                new_scale_name = weight_name + "_scale"
                new_state_dict[new_scale_name] = scale
                
                new_weight_map[weight_name] = file_name
                new_weight_map[new_scale_name] = file_name

            elif ("moe" in weight_name) and ("proj" in weight_name):
                
                i4_weights, scale, i8scales, i8zeros = weight_dequant_moe(weight, 128, group_size, format, symmetric, version)
                #scale:[num_experts, 1, out_feature]
                #i4_weights:[num_experts, out_feature//2, in_future]
                
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