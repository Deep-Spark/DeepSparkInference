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
import time
import urllib.error
import urllib.request

# Node ids in prompt.json; keep them in step with that file.
POSITIVE_PROMPT_NODE = "4"
SAMPLER_NODE = "7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Queue the MOSS-Image workflow on a running ComfyUI server"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188",
                        help="base URL of the ComfyUI server.")
    parser.add_argument("--workflow", default="prompt.json",
                        help="workflow in ComfyUI API format.")
    parser.add_argument("--prompt", default="",
                        help="override the positive prompt in the workflow.")
    parser.add_argument("--steps", type=int, default=0,
                        help="override the number of sampling steps.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="how long to wait for the run to finish, in seconds.")
    return parser.parse_args()


def post(server, path, payload):
    request = urllib.request.Request(
        f"{server}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def get(server, path):
    with urllib.request.urlopen(f"{server}{path}") as response:
        return json.load(response)


def main():
    args = parse_args()

    with open(args.workflow, encoding="utf-8") as handle:
        payload = json.load(handle)
    graph = payload.get("prompt", payload)
    if args.prompt:
        graph[POSITIVE_PROMPT_NODE]["inputs"]["text"] = args.prompt
    if args.steps:
        graph[SAMPLER_NODE]["inputs"]["steps"] = args.steps

    started = time.time()
    queued = post(args.server, "/prompt", {"prompt": graph})
    prompt_id = queued["prompt_id"]
    print(f"Queued {prompt_id}, waiting for the run to finish...")

    # /history only lists the run once ComfyUI has finished it, so poll until it shows up.
    while True:
        if time.time() - started > args.timeout:
            raise SystemExit(f"the run did not finish within {args.timeout}s")
        try:
            history = get(args.server, f"/history/{prompt_id}")
        except urllib.error.URLError as error:
            raise SystemExit(f"cannot reach the ComfyUI server: {error}")
        if prompt_id in history:
            break
        time.sleep(2)

    record = history[prompt_id]
    status = (record.get("status") or {}).get("status_str", "")
    if status and status != "success":
        raise SystemExit(f"the run failed: {json.dumps(record.get('status'), ensure_ascii=False)}")

    images = [
        image
        for output in (record.get("outputs") or {}).values()
        for image in (output.get("images") or [])
    ]
    if not images:
        raise SystemExit("the run produced no image")
    print(f"Generated in {time.time() - started:.1f}s")
    for image in images:
        print(f"  {image.get('subfolder', '')}/{image['filename']} ({image.get('type', 'output')})")


if __name__ == "__main__":
    main()
