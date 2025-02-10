import argparse
import os
import time

import text_generation_server.models as models
import torch
from text_generation_server.models.globals import set_adapter_to_index
from text_generation_server.pb import generate_pb2
from torch.cuda import profiler


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, default=None)
    parser.add_argument("--generate_length", type=int, default=512)
    parser.add_argument(
        "--model2path", type=str, default="/home/data/nlp/llama2/llama2-7b"
    )
    parser.add_argument("--speculate", type=int, default=0)
    parser.add_argument(
        "--quantize", type=str, default=None, choices=["awq", "bitsandbytes", "gptq"]
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    revision = None
    max_input_length = 1024
    max_prefill_tokens = 1024
    model_id = args.model2path
    test_model_name = model_id.split("/")
    set_adapter_to_index({})
    lora_adapter_ids = None
    model = models.get_model(
        model_id,
        lora_adapter_ids,
        revision,
        False,
        quantize=args.quantize,
        speculate=args.speculate,
        dtype=None,
        trust_remote_code=True,
        max_input_tokens=max_input_length,
    )

    if test_model_name[-1] == "":
        print(f"test_model_name: {test_model_name[-2]}")
    else:
        print(f"test_model_name: {test_model_name[-1]}")
    model_name = model_id.lower()
    if args.inputs:
        first_line = args.inputs
    else:
        if any(key in model_name for key in ["codellama", "flan-t5"]):
            first_line = "Tell me about AI"
        elif any(key in model_name for key in ["santacoder", "opt", "galactica"]):
            first_line = "Shanghai is one of the most prosperous cities in China, with a GDP of over $300 billion. Shanghai has the fastest growing economy in China and is the second busiest port in the world. In addition to being a hub for business, Shanghai is also a major tourist destination. It is known for its diverse culture and many historical sites.\nThe city of Shanghai is located on the coast of the Pacific Ocean in east-central China. It is bordered by Jiangsu Province to the north, Zhejiang Province to the south, and Jiangsu Province to the west."
        elif "mpt" in model_name:
            first_line = "Here is a recipe for vegan banana bread:\n"
        else:
            first_line = "蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是"

    default_pb_parameters = generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1,
        typical_p=1.0,
        do_sample=False,
    )

    default_pb_stop_parameters = generate_pb2.StoppingCriteriaParameters(
        stop_sequences=[], max_new_tokens=args.generate_length
    )

    warmup_requests = generate_pb2.Request(
        id=0,
        inputs="_test " * max_input_length,
        input_chunks=generate_pb2.Input(chunks=[generate_pb2.InputChunk(text="Test")]),
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
            ignore_eos_token=True,
        ),
        prefill_logprobs=True,
        top_n_tokens=512,
    )
    warmup_requests_batch = generate_pb2.Batch(id=0, requests=[warmup_requests], size=1)
    warmup_requests_batchs = model.batch_type.from_pb(
        warmup_requests_batch, model.tokenizer, model.dtype, torch.device("cuda")
    )

    model.warmup(warmup_requests_batchs)

    prompt_length = model.tokenizer(
        first_line, truncation=False, return_tensors="pt"
    ).input_ids[0]

    print(f"prompt length: {len(prompt_length)}")
    print(f"input text: {first_line}")
    pb_request = generate_pb2.Request(
        id=1,
        inputs=first_line,  # first_line
        input_chunks=generate_pb2.Input(
            chunks=[generate_pb2.InputChunk(text=first_line)]
        ),
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
        data = model.generate_token(next_batch_one)
    profiler.stop()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration_time = end_time - start_time
    generations_one = data[0]
    print(f"generate length: {generations_one[0].generated_text.generated_tokens}")
    print(f"one batch: {generations_one[0].generated_text.text}")
    print(f"qps: {generations_one[0].generated_text.generated_tokens / duration_time}")