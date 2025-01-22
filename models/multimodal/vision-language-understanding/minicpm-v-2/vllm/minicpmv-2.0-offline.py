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

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

def main(args):
    # 图像文件路径列表
    ## wget https://img.zcool.cn/community/012e285a1ea496a8012171323c6bf1.jpg@3000w_1l_0o_100sh.jpg -O dog.jpg
    IMAGES = [
        args.image_path,  # 本地图片路径
    ]

    # 模型名称或路径
    MODEL_NAME = args.model_path  # 本地模型路径或Hugging Face模型名称

    # 打开并转换图像
    image = Image.open(IMAGES[0]).convert("RGB")

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 初始化语言模型
    llm = LLM(model=MODEL_NAME,
            gpu_memory_utilization=0.95,  # 使用全部GPU内存
            trust_remote_code=True,
            max_model_len=2048,
            # max_num_seqs=1,
            max_num_batched_tokens=2048,)  # 根据内存状况可调整此值

    # 构建对话消息
    messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '请描述这张图片'}]

    # 应用对话模板到消息
    prompt = tokenizer.apply_chat_template(messages)

    # 设置停止符ID
    # 2.0
    stop_token_ids = [tokenizer.eos_id]
    # 2.5
    #stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
    # 2.6 
    # stop_tokens = ['<|im_end|>', '<|endoftext|>']
    # stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    # 设置生成参数
    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids,
        # temperature=0.7,
        # top_p=0.8,
        # top_k=100,
        # seed=3472,
        max_tokens=1024,
        # min_tokens=150,
        temperature=0,
        # use_beam_search=False,
        # length_penalty=1.2,
        best_of=1)

    # 获取模型输出
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        }
    }, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="model path")
    parser.add_argument("--image-path", type=str, default=None, help="sample image path")
    args = parser.parse_args()

    main(args)