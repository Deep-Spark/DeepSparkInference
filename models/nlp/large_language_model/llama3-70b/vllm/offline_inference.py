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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import load_chat_template,sampling_add_cli_args

import logging
import time
import argparse
import dataclasses
import inspect

import torch
from vllm import LLM, SamplingParams, EngineArgs


parser = argparse.ArgumentParser()
parser.add_argument("--qps_test", default=False, action='store_true', help="for test only!!")
parser.add_argument("--acc_test", default=False, action='store_true', help="for test only!!")
parser.add_argument("--qps_threshold", type=float, default=15., help="for test only!!")
parser.add_argument("--acc_threshold", type=float, default=0.95, help="for test only!!")
parser.add_argument("--chat_template",type=str,default=None)
parser.add_argument("--remove_chat_template",default=False,action="store_true",help="pass this if you are not use a chat model")
parser = EngineArgs.add_cli_args(parser)
parser = sampling_add_cli_args(parser)
args = parser.parse_args()

engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
sampling_args = [param.name for param in list(inspect.signature(SamplingParams.__init__).parameters.values())[1:]]
engine_params = {attr:getattr(args, attr) for attr in engine_args}
sampling_params = {attr:getattr(args, attr) for attr in sampling_args if args.__contains__(attr)}

model_name = args.model.strip()
model_name = model_name if args.model[-1]!='/' else model_name[:-1]
model_name = model_name.rsplit('/')[-1]


# Sample prompts.
if not args.qps_test and not args.acc_test:
    prompts = [
                "Shanghai is one of the most prosperous cities in China, with a GDP of over $300 billion. Shanghai has the fastest growing economy in China and is the second busiest port in the world. In addition to being a hub for business, Shanghai is also a major tourist destination. It is known for its diverse culture and many historical sites.\nThe city of Shanghai is located on the coast of the Pacific Ocean in east-central China. It is bordered by Jiangsu Province to the north, Zhejiang Province to the south, and Jiangsu Province to the west.", 
                "What signs may indicate that a person is experiencing anxiety?", 
                "Describe how to make cheese pizza.", 
                "Write a review article on the development of 5G networks."
            ]
else:
    prompts = ["Shanghai is one of the most prosperous cities in China, with a GDP of over $300 billion. Shanghai has the fastest growing economy in China and is the second busiest port in the world. In addition to being a hub for business, Shanghai is also a major tourist destination. It is known for its diverse culture and many historical sites.\nThe city of Shanghai is located on the coast of the Pacific Ocean in east-central China. It is bordered by Jiangsu Province to the north, Zhejiang Province to the south, and Jiangsu Province to the west.",]

# Create a sampling params object.
sampling_params = SamplingParams(**sampling_params)

# Create an LLM.
llm = LLM(**engine_params)

# process chat template
if args.remove_chat_template:
    if 'chat' in model_name.lower():
        logging.warning(f"The model name from model path is {model_name}, so we guess you are using the chat model and the additional processing is required for the input prompt. "
                        f"If the result is not quite correct, please ensure you do not pass --remove_chat_template in CLI.")
    prompts_new = prompts
else:
    # Build chat model promopt
    logging.warning("If you are using a non chat model, please pass the --remove_chat_template in CLI.")
    # Try use transformers's apply_chat_template, if chat_template is None, will use defalut template.
    # For some old models, the default template may cause bad answers. we don't consider this situation, 
    # because the Transformers team is advancing the chat template. For more informatino about it, 
    # please refer to https://huggingface.co/docs/transformers/main/chat_templating
    try:
        load_chat_template(llm.get_tokenizer(),args.chat_template)
        prompts_new = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = llm.get_tokenizer().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts_new.append(text)
    except:
        logging.warning("use tokenizer apply_chat_template function failed, may because of low transformers version...(try use transformers>=4.34.0)")
        prompts_new = prompts

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts_new, sampling_params,use_tqdm=False) if isinstance(prompts_new[0],str) else llm.generate(sampling_params=sampling_params,prompt_token_ids=prompts_new,use_tqdm=False)
torch.cuda.synchronize()

start_time = time.perf_counter()
outputs = llm.generate(prompts_new, sampling_params) if isinstance(prompts_new[0],str) else llm.generate(sampling_params=sampling_params,prompt_token_ids=prompts_new)
torch.cuda.synchronize()
end_time = time.perf_counter()
duration_time = end_time - start_time

num_tokens = 0
# Print the outputs.
for i, output in enumerate(outputs):
    prompt = prompts[i] # show the origin prompt. actully prompt is "output.prompt"
    generated_text = output.outputs[0].text
    
    num_tokens += len(output.outputs[0].token_ids)
    # if not args.qps_test and not args.acc_test:
    print(f"Prompt: {prompt}\nGenerated text: {generated_text} \n")
print(f"tokens: {num_tokens}, QPS: {num_tokens/duration_time}")

# test use
if args.qps_test:
    import re
    qps_dicts = {'llama.*7.*' :{1:36, 2:42, 8:30},
                 'llama.*13.*':{1:20, 2:28, 8:30}
                 }
    for k,v in qps_dicts.items():
        if re.search(k,model_name.lower()):
            qps_dict = v
            break
    args.qps_threshold = qps_dict.get(args.tensor_parallel_size,4.5)
    if num_tokens/duration_time < args.qps_threshold:
        print('val qps: {}, target qps: {}, fail'.format(num_tokens/duration_time,args.qps_threshold))
        exit(1)
    print('val qps: {}, target qps: {}, pass'.format(num_tokens/duration_time,args.qps_threshold))
if args.acc_test:
    from rouge import Rouge
    import re
    acc_dict  = {r'llama.?7.*?':"",
                 r'llama.?13.*?':" Shanghai is located on the Yangtze River Delta, which is the largest river delta in the world.\nShanghai has a humid subtropical climate with four distinct seasons. The summers are hot and humid, with temperatures reaching 35 degrees Celsius. The winters are cool and dry, with temperatures reaching 10 degrees Celsius. The city receives an average of 1,200 millimeters of rain per year.\nShanghai is the most populous city in China, with a population of over 23 million. The city is home to 9 million permanent residents and 14 million migrant workers. Shanghai is also home to the largest number of expatriates in China.\nShanghai is a major tourist destination, with over 23 million visitors per year. The city is home to many historical sites, including the Bund, the Yu Garden, and the Jade Buddha Temple. Shanghai is also home to the Oriental Pearl TV Tower, which is the second tallest tower in the world.\nShanghai is a major center for business and finance in China. It is home to the Shanghai Stock Exchange,",
                 r'llama.?2.?7.*?':" The city has a population of over 24 million people and covers an area of 6,340 square kilometers.\nShanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites.\nShanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites.\nShanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites. Shanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites.\nShanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites. Shanghai is a major port city and is home to many large companies. The city is also a major tourist destination and is known for its many historical sites. Shanghai is a major port city and is home to many large companies. The city is also a major tourist",
                 r'llama.?2.?13.*?':" The city has a population of over 24 million people and covers an area of 6,340 square kilometers (2,448 square miles).\nShanghai is a major financial center in China and is home to many multinational corporations. The city has a diverse economy that includes manufacturing, finance, real estate, and tourism. Shanghai is also a major transportation hub with two international airports and a large port.\nThe city of Shanghai is divided into 16 districts. The districts are:\n1. Huangpu District\n2. Xuhui District\n3. Changning District\n4. Jing’an District\n5. Putuo District\n6. Yangpu District\n7. Hongkou District\n8. Baoshan District\n9. Minhang District\n10. Jiading District\n11. Qingpu District\n12. Songjiang District\n13. Fengxian District\n14. Jinshan District\n15. Nanhui District\n16. Pudong New Area\nThe city of Shanghai is divided into 16 districts. The districts are: Huangpu District, Xu",
                }
    for k, v in acc_dict.items():
        if re.search(k,model_name.lower()):
            reference = v
            break
    candidate = outputs[0].outputs[0].text
    scorer = Rouge()
    scores = scorer.get_scores(reference, candidate)
    if scores[0]["rouge-1"]['f'] < args.acc_threshold:
        print('val ROUGE-1 score f1: {}, target ROUGE-1 score f1: {}, fail'.format(scores[0]["rouge-1"]['f'],args.acc_threshold))
        exit(1)
    print('val ROUGE-1 score f1: {}, target ROUGE-1 score f1: {}, pass'.format(scores[0]["rouge-1"]['f'],args.acc_threshold))
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["val ROUGE-1 score f1"] = scores[0]["rouge-1"]['f']
    metricResult["metricResult"]["val ROUGE-1 score f1"] = args.acc_threshold
    print(metricResult)

# 2 7b vllm 0.1.6: batch 3, tokens: 773, QPS: 64.35866137433203; batch 1, tokens: 257, QPS: 25.396898421442113
# 1\2 13b vllm 0.1.6: batch 3, tokens: 768, QPS: 41.538942353799506; batch 1, tokens: 257, QPS: 15.639606595029639 (2, 6.5829828847570795; 8, 5.137610167755676)

# 0.3.2 13b tokens: 768, QPS: 99.1182273040533 13b-awq-2card(tokens: 768, QPS: 161.07526866069998) 70b-awq-2card(tokens: 768, QPS: 55.91434180918294)
# 0.3.2 smoothquant 7b tokens: 750, QPS: 82.11710297948171(tokens: 768, QPS: 82.49768795244577)