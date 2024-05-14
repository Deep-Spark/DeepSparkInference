from text_generation_server.models.flash_qwen import (
        FlashQwen,
    )
import torch
from text_generation_server.pb import generate_pb2

import time
from torch.cuda import profiler
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_length', type=int, default=512)
    parser.add_argument('--model2path', type=str, default="/home/data/nlp/qwen/qwen-7B")
    parser.add_argument('--quantize', type=str, default=None, choices=['awq'])
    parser.add_argument('--speculate', type=int, default=0)

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    isNewVersion = True
    try:
        from text_generation_server.utils.speculate import set_speculate
    except ImportError:
        isNewVersion = False
        print("use n-gram speculate must update tgi version to 1.4.3+")
    else:
        set_speculate(args.speculate)
    max_input_length = 2048
    max_prefill_tokens = 2048
    model = FlashQwen(args.model2path, trust_remote_code=True)

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
        data = model.generate_token(next_batch_one)
        if isNewVersion:
            generations_one, next_batch_one, _ = data
        else:
            generations_one, next_batch_one = data
        if next_batch_one is None:
            last_generations = False
            break
    if last_generations:
        data = model.generate_token(next_batch_one)
    generations_one = data[0]
    profiler.stop()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time
    print(f"generate length: {generations_one[0].generated_text.generated_tokens}")
    print(f"one batch: {generations_one[0].generated_text.text}\nqps: {generations_one[0].generated_text.generated_tokens /duration_time}")

"""
qwen-7B
亚的斯亚贝巴（Addis Ababa）
尼日利亚的首都是阿布贾（Abuja）
巴基斯坦的首都是伊斯兰堡（Islamabad）
菲律宾的首都是马尼拉（Manila）
波兰的首都是华沙（Warsaw）
葡萄牙的首都是里斯本（Lisbon）
俄罗斯的首都是莫斯科（Moscow）
新加坡的首都是新加坡（Singapore）
南非的首都是比勒陀利亚（Pretoria）
西班牙的首都是马德里（Madrid）
斯里兰卡的首都是斯里贾亚瓦德纳普拉克特（Sri Jayawardenepura Kotte）
斯洛伐克的首都是布拉迪斯拉发（Bratislava）
斯洛文尼亚的首都是卢布尔雅那（Ljubljana）
南非的首都是比勒陀利亚（Pretoria）
瑞典的首都是斯德哥尔摩（Stockholm）
瑞士的首都是伯尔尼（Bern）
泰国的首都是曼谷（Bangkok）
土耳其的首都是安卡拉（Ankara）
乌克兰的首都是基辅（Kyiv）
英国的首都是伦敦（London）
美国的首都是华盛顿特区（Washington, D.C.）
乌兹别克斯坦的首都是塔什干（Tashkent）
委内瑞拉的首都是加拉加斯（Caracas）
越南的首都是河内（Hanoi）
赞比亚的首都是卢萨卡（Lusaka）
津巴布韦的首都是哈拉雷（Harare）
以上是世界上一些国家的首都，当然还有很多其他国家的首都，这里只是列举了一些比较有代表性的。  2022年广东省公务员考试公告于11月26日发布，报考者可在 2021年11月29日9︰00至12月3日16︰00 的时间内报名。建议小伙伴们根据本人的专业、意愿和职业规划等选择报考职位，不要等到最后才匆忙报名，以免因时间不足等情况无法完成报名而造成遗憾。
   ——2022年广东省考报名有关解答——
  报考者如何办理考试费减免手续?
  答：报考者如属城乡最低生活保障对象，可向报考职位所在考区考务机构申请减免考试费，申请对象需提交其家庭所在地的县(区、
qps: 34.23966521171583
"""