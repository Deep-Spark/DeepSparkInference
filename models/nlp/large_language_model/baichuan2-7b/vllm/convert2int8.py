import argparse
import glob
import json
import os
import shutil
import sys
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
args = parser.parse_args()


def float2int8(load_path, save_path):
    all_files = glob.glob(os.path.join(load_path, "*"))
    os.makedirs(save_path)
    print(f"save int8 weight to: {save_path}")
    for raw_file in all_files:
        ext_name = os.path.splitext(raw_file)[-1]
        if ext_name in [".json", ".py", ".model"]:
            dst_file = os.path.split(raw_file)[-1]
            dst_file = os.path.join(save_path, dst_file)
            shutil.copy(raw_file, dst_file)
            print(f"copy file `{raw_file}` to `{dst_file}`")
        elif ext_name == ".bin":
            print(f"quantize `{raw_file}`")
            params = torch.load(raw_file, map_location="cpu")
            new_params = OrderedDict()
            keys = ["proj", "pack"]
            for k, v in params.items():
                find_key = False
                for key in keys:
                    if key in k:
                        scale = torch.abs(v).max(dim=-1)[0] / 127.0
                        int8_v = (
                            torch.clamp(v / scale.view(-1, 1), min=-127, max=127)
                            .to(torch.int8)
                            .contiguous()
                        )
                        scale = scale.view(1, -1).contiguous()
                        new_params[k] = int8_v
                        new_params[k.replace("weight", "scales")] = scale
                        find_key = True
                        break
                if find_key:
                    continue
                # save the other param
                new_params[k] = v
            file_name = os.path.basename(raw_file)
            file_name_no_suffix = file_name.rsplit(".", 1)[0]
            new_file_name = file_name_no_suffix + "_int8.bin"
            torch.save(new_params, os.path.join(save_path, new_file_name))

    config_file = os.path.join(save_path, "w8a16_config.json")
    with open(config_file, "w") as f:
        f.write(json.dumps({}))


if __name__ == "__main__":
    model_path = args.model_path
    save_path = os.path.join(model_path, "int8")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    float2int8(model_path, save_path)
