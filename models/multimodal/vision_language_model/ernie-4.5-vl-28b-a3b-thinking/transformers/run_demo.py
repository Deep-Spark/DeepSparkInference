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

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

model_path = 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model.add_image_preprocess(processor)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What color clothes is the girl in the picture wearing?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                }
            },
        ]
    },
]

text = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

device = next(model.parameters()).device
inputs = inputs.to(device)

generated_ids = model.generate(
    inputs=inputs['input_ids'].to(device),
    **inputs,
    max_new_tokens=1024,
    use_cache=False
    )
output_text = processor.decode(generated_ids[0][len(inputs['input_ids'][0]):])
print(output_text)
