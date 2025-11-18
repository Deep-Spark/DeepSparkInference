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

import argparse
import time
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

def main(args):
    model_path = args.model_path
    max_new_tokens = args.max_tokens

    backend_config = PytorchEngineConfig(session_len=2048, tp=args.tp)
    gen_config = GenerationConfig(top_p=0.8,
                                  top_k=40,
                                  temperature=0.8,
                                  max_new_tokens=max_new_tokens)

    pipe = pipeline(model_path,
                    backend_config=backend_config)
    prompts = [[{
        'role': 'user',
        'content': '请介绍一下你自己'
    }], [{
        'role': 'user',
        'content': '请介绍一下上海'
    }]]
    start_time = time.perf_counter()
    response = pipe(prompts, gen_config=gen_config)
    end_time = time.perf_counter()
    duration_time = end_time - start_time

    num_tokens = 0
    # Print the outputs.
    for i, output in enumerate(response):
        prompt = prompts[i]  # show the origin prompt. actully prompt is "output.prompt"
        generated_text = output.text
        num_tokens += len(output.token_ids)
        print(f"Prompt: {prompt}\nGenerated text: {generated_text} \n")
    num_requests = len(prompts)  # 请求的数量
    qps = num_requests / duration_time
    print(f"requests: {num_requests}, QPS: {qps}, tokens: {num_tokens}, Token/s: {num_tokens/duration_time}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)

    args = parser.parse_args()
    main(args)

