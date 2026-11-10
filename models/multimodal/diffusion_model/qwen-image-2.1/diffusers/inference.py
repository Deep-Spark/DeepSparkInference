# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import json
import os
import time

import numpy as np
import torch
import diffusers
import transformers
from diffusers import DiffusionPipeline

DEFAULT_PROMPT = (
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, '
    "reflections on wet pavement"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 text-to-image inference")
    parser.add_argument("--model", required=True, help="path to the model directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt.")
    parser.add_argument("--steps", type=int, default=20, help="number of denoising steps.")
    parser.add_argument("--size", type=int, default=1024, help="width and height of the image.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument("--output", default="generated.png", help="where to save the image.")
    return parser.parse_args()


def custom_pipeline(model_path):
    """A pipeline class diffusers does not ship comes in a .py beside model_index.json.

    Loading it means running that file, which is what the model card asks for.
    """
    with open(os.path.join(model_path, "model_index.json"), encoding="utf-8") as handle:
        declared = json.load(handle).get("_class_name") or ""
    name = declared[-1] if isinstance(declared, list) else str(declared)
    if not name or hasattr(diffusers, name):
        return {}
    for item in sorted(os.listdir(model_path)):
        path = os.path.join(model_path, item)
        if item.endswith(".py") and os.path.isfile(path):
            with open(path, encoding="utf-8", errors="ignore") as handle:
                if "class " + name in handle.read():
                    return {"custom_pipeline": path, "trust_remote_code": True}
    raise SystemExit(
        f"model_index.json declares pipeline {name}, which is neither in diffusers "
        "nor defined by a .py in the model directory"
    )


def main():
    args = parse_args()

    started = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        args.model, dtype=torch.bfloat16, **custom_pipeline(args.model)
    )
    # The pipeline is larger than one card, so components move on and off the GPU as
    # they are used. This trades throughput for fit.
    pipe.enable_model_cpu_offload()
    loaded = time.time() - started
    print(f"diffusers {diffusers.__version__}, transformers {transformers.__version__}, torch {torch.__version__}")
    print(f"Pipeline {type(pipe).__name__} loaded in {loaded:.1f}s")

    started = time.time()
    image = pipe(
        prompt=args.prompt,
        width=args.size,
        height=args.size,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    elapsed = time.time() - started

    image.save(args.output)
    # A pipeline that silently produced a flat or pure-noise canvas still "returns an
    # image", so check the pixels rather than the call succeeding.
    spread = float(np.asarray(image.convert("RGB")).astype("float32").std())
    print(f"Generated {image.size[0]}x{image.size[1]} in {elapsed:.1f}s ({args.steps} steps)")
    print(f"Pixel std {spread:.1f}, saved to {args.output}")
    if list(image.size) != [args.size, args.size] or spread <= 5.0:
        raise SystemExit("the generated image is blank or pure noise")


if __name__ == "__main__":
    main()
