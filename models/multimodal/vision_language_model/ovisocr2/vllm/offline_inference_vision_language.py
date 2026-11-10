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

from vllm import LLM, SamplingParams

from utils import image_payload


def main(args):
    # LLM.chat applies the checkpoint's own chat template and decodes the image the
    # same way the HTTP server does, so an offline run and an online one see the same
    # prompt. LLM.generate would hand the model a bare string with no image slot.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_payload(args.image)}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    start = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    print(f"\nModel loaded in {time.time() - start:.1f}s")

    start = time.time()
    outputs = llm.chat(
        [conversation] * args.num_prompts,
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )
    elapsed = time.time() - start

    tokens = 0
    for output in outputs:
        tokens += len(output.outputs[0].token_ids)
        print("-" * 20 + " Generated Text " + "-" * 20)
        print(output.outputs[0].text)
        print("-" * 56)
    print(
        f"requests: {args.num_prompts}, QPS: {args.num_prompts / elapsed:.2f}, "
        f"tokens: {tokens}, Token/s: {tokens / elapsed:.2f}"
    )


if __name__ == "__main__":
    # Tensor parallelism spawns worker processes that re-import this file, so the
    # engine has to stay behind this guard or every worker would build its own.
    parser = argparse.ArgumentParser(
        description="vLLM OvisOCR2 offline inference"
    )

    # 模型配置 |Model configuration
    parser.add_argument("--model", type=str, default="./data/OvisOCR2",
                        help="Path to the model checkpoint.")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of GPUs to shard the model over.")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum context length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of each GPU reserved for the engine.")
    # Eager is the default because graph capture currently fails on this stack, but
    # --no-enforce-eager has to remain reachable so the graph path can be retried.
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Disable torch.compile and CUDA graph capture.")

    # 推理参数 |Reasoning parameters
    # OvisOCR2 parses document pages. Given a photograph it has no text to read and
    # loops on filler until max_tokens, so the default asset is a page, not a scene.
    parser.add_argument("--image", type=str,
                        default="./assets/document_page.png",
                        help="Local path or URL of the page image to parse.")
    parser.add_argument("--prompt", type=str,
                        default="Parse this document page into Markdown.",
                        help="Prompt text.")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="Number of identical requests to run.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens to generate.")

    main(parser.parse_args())
