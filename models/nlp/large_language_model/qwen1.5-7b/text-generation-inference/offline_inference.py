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

from text_generation_server.models.flash_qwen2 import (
    FlashQwen2,
)
import torch
from text_generation_server.pb import generate_pb2

import time
from torch.cuda import profiler
from text_generation_server.utils.speculate import set_speculate
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_length', type=int, default=512)
    parser.add_argument('--model2path', type=str, default="/home/data/nlp/qwen2/Qwen1.5-0.5B")
    parser.add_argument('--quantize', type=str, default=None, choices=['awq'])
    parser.add_argument('--speculate', type=int, default=0)

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()

    max_input_length = 2048
    max_prefill_tokens = 2048

    set_speculate(args.speculate)
    model = FlashQwen2(args.model2path, trust_remote_code=True, quantize=args.quantize)

    first_line = "蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是"

    default_pb_parameters = generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1,
        typical_p=1.0,
        do_sample=False,
    )
    
    default_pb_stop_parameters = generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=args.generate_length)
    
    warmup_requests =  generate_pb2.Request(
        id=0,
        inputs="_test " * max_input_length,
        prefill_logprobs=True,
        truncate=max_input_length,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            do_sample=False,
            seed=0,
            repetition_penalty=1.2,
            watermark=True,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=2,
            stop_sequences=[],
            ignore_eos_token=False,
        ),
        top_n_tokens = 20
    )
    warmup_requests_batch = generate_pb2.Batch(id=0, requests=[warmup_requests], size=1)
    warmup_requests_batchs =  model.batch_type.from_pb(
        warmup_requests_batch, model.tokenizer, model.dtype, torch.device("cuda")
    )
    
    model.warmup(warmup_requests_batchs)

    pb_request = generate_pb2.Request(
        id=1,
        inputs=first_line,
        prefill_logprobs=True,
        truncate=1024,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )
    pb_one_batch = generate_pb2.Batch(id=1, requests=[pb_request], size=1)
    causal_lm_one_batch = model.batch_type.from_pb(
        pb_one_batch, model.tokenizer, model.dtype, torch.device("cuda")
    )

    next_batch_one = causal_lm_one_batch
    last_generations = True 
    torch.cuda.synchronize()
    profiler.start()
    start_time = time.perf_counter()
    for _ in range(causal_lm_one_batch.stopping_criterias[0].max_new_tokens - 1):
        generations_one, next_batch_one, _ = model.generate_token(next_batch_one)
        if next_batch_one is None:
            last_generations = False
            break
    if last_generations:
        generations_one, next_batch_one, _ = model.generate_token(next_batch_one)
    profiler.stop()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time
    print(f"generate length: {generations_one[0].generated_text.generated_tokens}")
    print(f"one batch: {generations_one[0].generated_text.text}\nqps: {generations_one[0].generated_text.generated_tokens /duration_time}")
    metricResult = {"metricResult": {}}
    metricResult["metricResult"]["generate length"] = generations_one[0].generated_text.generated_tokens
    metricResult["metricResult"]["one batch"] = generations_one[0].generated_text.text
    metricResult["metricResult"]["qps"] = generations_one[0].generated_text.generated_tokens /duration_time
    print(metricResult)

"""
qwen1.5-0.5B
one batch: 亚历山大（Alexandria）
俄罗斯的首都是莫斯科（Moscow）
土耳其的首都是伊斯坦布尔（Istanbul）
南非的首都是开普敦（Cape Town）
美国的首都是华盛顿（Washington）
澳大利亚的首都是堪培拉（Canberra）
印度的首都是新德里（New Delhi）
法国的首都是巴黎（Paris）
英国的首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（Paris）
英国首都是伦敦（London）
加拿大首都是温哥华（Vancouver）
南非首都是开普敦（Cape Town）
美国首都是华盛顿（Washington）
澳大利亚首都是堪培拉（Canberra）
印度首都是新德里（New Delhi）
法国首都是巴黎（
qps: 128.489649542011
"""