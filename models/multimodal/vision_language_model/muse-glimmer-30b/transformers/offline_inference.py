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
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Muse-Glimmer-30B offline vision-language inference"
    )
    parser.add_argument("--model", required=True, help="path to the model directory.")
    parser.add_argument("--image", required=True, help="path to the input image.")
    parser.add_argument("--prompt", default="Describe this image in one sentence.",
                        help="question asked about the image.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="maximum number of tokens to generate.")
    return parser.parse_args()


def memory_budget():
    """Leave room for activations on every card; what still does not fit goes to host memory.

    The BF16 weights are about 55.5 GiB, so on two 32 GiB cards some modules are
    offloaded and the latency then includes moving them in and out. The host figure
    has to be measured rather than assumed: it is the only thing telling accelerate
    whether an offloaded module fits in memory or has to spill to disk, and claiming
    more than the machine has turns a slow run into an OOM kill.
    """
    cards = torch.cuda.device_count()
    if not cards:
        raise SystemExit("no GPU is visible")
    budget = {
        index: "%dGiB" % int(
            torch.cuda.get_device_properties(index).total_memory / 2**30 * 0.88
        )
        for index in range(cards)
    }
    host = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    budget["cpu"] = "%dGiB" % int(host * 0.7)
    on_card = sum(
        torch.cuda.get_device_properties(index).total_memory for index in range(cards)
    ) / 2**30
    if host * 0.7 + on_card < 60:
        print(
            f"WARNING: {cards} card(s) plus {host}GiB of host memory may not hold the "
            "55.5GiB of BF16 weights; the run can fail on allocation"
        )
    return budget


def main():
    args = parse_args()

    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", max_memory=memory_budget()
    ).eval()
    print(f"Model loaded in {time.time() - started:.1f}s")

    placement = getattr(model, "hf_device_map", None) or {}
    devices = sorted({str(device) for device in placement.values()})
    offloaded = sum(1 for device in placement.values() if str(device) in ("cpu", "disk"))
    print(f"Devices: {', '.join(devices) or 'unknown'}")
    if offloaded:
        print(f"WARNING: {offloaded} modules did not fit the GPUs and run from host memory")

    if not os.path.isfile(args.image):
        raise SystemExit(
            f"image not found: {args.image}. The README copies the shared assets with "
            "`cp -r ../../vllm_public_assets/ ./` before running this."
        )
    # convert() returns a new image, so the file handle can close right away.
    with Image.open(args.image) as handle:
        picture = handle.convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": picture},
        {"type": "text", "text": args.prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    elapsed = time.time() - started

    answer = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
    print("-" * 20 + " Generated Text " + "-" * 20)
    print(answer)
    print("-" * 56)
    print(
        f"Prompt tokens {prompt_tokens}, "
        f"output tokens {int(output.shape[-1]) - prompt_tokens}, "
        f"latency {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
