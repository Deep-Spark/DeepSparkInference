# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import argparse
import time
from openai import OpenAI
from transformers import AutoTokenizer


def send_request(
    api_url: str,
    prompt: str,
    output_len: int,
    stream: bool,
) -> None:
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="EMPTY",
        base_url=api_url,
    )

    models = client.models.list()
    model = models.data[0].id
    
    completion = client.completions.create(
        model=model,
        # messages=[{"role": "user", "content": prompt},],
        prompt=prompt,
        n=1,
        stream=stream,
        max_tokens=output_len, 
        temperature=0.0
    )
    
    if stream:
        for each_com in completion:
            print(each_com)
    else:
        print("++++++++++++++++++")
        print(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output_token", type=int, default=1024)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/v1"
    
    prompts = [
            "你好",
            "Which city is the capital of China?",
            "1 + 1 = ?",
            "中国的首都是哪里", 
            "请讲以下内容翻译为英文：\n你好,我来自中国。",
            ]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [tokenizer.build_chat_input(i).input_ids.tolist() for i in prompts]

    for prompt in prompts:
        send_request(api_url,prompt,args.output_token,args.stream)
