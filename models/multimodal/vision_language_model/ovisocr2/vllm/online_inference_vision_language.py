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
import base64
import mimetypes
import os
import time

from openai import OpenAI


def image_payload(source):
    """A local page scan has to be inlined; an http(s) address is passed through."""
    if source.startswith(("http://", "https://", "data:")):
        return source
    if not os.path.isfile(source):
        raise SystemExit(f"image not found: {source}")
    mime = mimetypes.guess_type(source)[0] or "image/png"
    with open(source, "rb") as handle:
        return f"data:{mime};base64," + base64.b64encode(handle.read()).decode()


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_payload(args.image)}},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    print(f"Sending request to {args.base_url} with model {args.model_name}...")
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as error:
        print(f"Error during inference: {error}")
        return

    elapsed = time.time() - start
    print(f"\nResponse costs: {elapsed:.2f}s")
    print("-" * 20 + " Generated Text " + "-" * 20)
    print(response.choices[0].message.content)
    print("-" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client for vLLM OvisOCR2 Online Inference"
    )

    # 服务器配置 |Server configuration
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM API server base URL.")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="API key for the server.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Request timeout in seconds.")

    # 模型配置 |Model configuration
    parser.add_argument("--model-name", type=str, default="OvisOCR2",
                        help="Name of the model served by vLLM.")

    # 推理参数 |Reasoning parameters
    # OvisOCR2 parses document pages. Given a photograph it has no text to read and
    # loops on filler until max_tokens, so the default asset is a page, not a scene.
    parser.add_argument("--image", type=str,
                        default="./assets/document_page.png",
                        help="Local path or URL of the page image to parse.")
    parser.add_argument("--prompt", type=str,
                        default="Parse this document page into Markdown.",
                        help="Prompt text.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum tokens to generate.")

    main(parser.parse_args())
