#!/bin/bash
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
#
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams
from PIL import Image 
import argparse

def inference(args):
    # prepare model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size = 2,
        pipeline_parallel_size  = 4,
        limit_mm_per_prompt={"image": 5},
    )
    
    # prepare inputs
    question = "请描述这张图片"
    
    image = Image.open("./vllm_public_assets/cherry_blossom.jpg")
    image = image.convert("RGB")
    inputs = {
        # "prompt": f"<|user|>\n<|image|>\n{question}<|end|>\n<|assistant|>\n",
        "prompt": f"<s>[INST]{question}\n[IMG][/INST]",
        "multi_modal_data": {
            "image": image
        },
    }
    
    # generate response
    print("========== SAMPLE GENERATION ==============")
    outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=1024))
    print(f"RESPONSE: {outputs[0].outputs[0].text}")
    print("==========================================")
    
    
def main():
    parser = argparse.ArgumentParser(description="Example script with --model and --port arguments")
    parser.add_argument("--model", type=str, default="/data/nlp/Pixtral-Large-Instruct-2411-hf-quantized.w4a16/", help="Model name or path")
    args = parser.parse_args()
    
    
    inference(args)

 
    
if __name__ == "__main__":
    main()    
