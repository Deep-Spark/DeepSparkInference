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
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig

def main(args):
    model_path = args.model_path
    max_new_tokens = args.max_tokens

    backend_config = PytorchEngineConfig(session_len=2048,tp = args.tp)
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
    response = pipe(prompts, gen_config=gen_config)
    print(response)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)

    args = parser.parse_args()
    main(args)

