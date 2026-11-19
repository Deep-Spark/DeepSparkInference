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
import time

import torch
from diffusers import QwenImage21Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./data/Qwen-Image-2.1-Turbo")
    parser.add_argument(
        "--prompt",
        default='A neon shop sign that reads "QWEN IMAGE 2.1 TURBO", rainy night, reflections on wet pavement',
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="generated.png")
    args = parser.parse_args()

    started = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    # Pipeline is larger than one BI-V150 card; offload components as needed.
    pipe.enable_model_cpu_offload()
    print(f"loaded in {time.time() - started:.1f}s")

    started = time.time()
    image = pipe(
        prompt=args.prompt,
        width=args.size,
        height=args.size,
        use_kv_cache=True,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    print(f"generated in {time.time() - started:.1f}s")

    image.save(args.output)
    print(f"saved {args.output} size={image.size} mode={image.mode}")


if __name__ == "__main__":
    main()
