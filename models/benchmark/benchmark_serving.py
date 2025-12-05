import argparse
import asyncio
import json
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from vllm import AsyncEngineArgs, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


@dataclass
class RequestFuncInput:
    prompt: List[int]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


def get_random_lens(
    input_tokens,
    output_tokens,
    max_input_tokens,
    max_output_tokens,
    num_requests,
    input_output_type,
):
    assert 1 <= input_tokens < max_input_tokens
    assert 1 <= output_tokens < max_output_tokens
    min_input_tokens = input_tokens
    min_output_tokens = output_tokens
    input_mean = int((max_input_tokens + min_input_tokens) / 2)
    input_std = int((max_input_tokens - input_mean) / 2)
    output_mean = int((max_output_tokens + min_output_tokens) / 2)
    output_std = int((max_output_tokens - output_mean) / 2)

    input_len_list = []
    output_len_list = []
    for _ in range(num_requests):
        if input_output_type == "normal":
            while True:
                input_length = int(np.random.normal(input_mean, input_std))
                if min_input_tokens <= input_length <= max_input_tokens:
                    break
            while True:
                output_length = int(np.random.normal(output_mean, output_std))
                if min_output_tokens <= output_length <= max_output_tokens:
                    break
        else:
            input_length = int(np.random.uniform(min_input_tokens, max_input_tokens))
            output_length = int(np.random.uniform(min_output_tokens, max_output_tokens))

        input_len_list.append([None, input_length])
        output_len_list.append(output_length)

    return input_len_list, output_len_list


def sample_requests(
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    input_output_type: str = "fix",
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[int]]:
    np.random.seed(seed)
    random.seed(seed)

    if input_output_type == "fix":
        input_len_list = [[None, input_tokens] for _ in range(num_requests)]
        output_len_list = [output_tokens for _ in range(num_requests)]
    elif input_output_type in ["normal", "uniform"]:
        input_len_list, output_len_list = get_random_lens(
            input_tokens,
            output_tokens,
            max_input_tokens,
            max_output_tokens,
            num_requests,
            input_output_type,
        )
    else:
        raise NotImplementedError("You can modify this code according to your needs")

    for inputs in input_len_list:
        assert len(inputs) == 2
        # [str, int] or [None ,int]
        assert isinstance(inputs[0], str) or inputs[0] is None
        assert isinstance(inputs[1], int)
    for outptus in output_len_list:
        assert isinstance(outptus, int)

    return input_len_list, output_len_list


async def get_request(
    input_requests: List[RequestFuncInput],
    time_interval: float,
):
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request
        
        if time_interval == 0:
            continue
        await asyncio.sleep(time_interval)


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        headers = {
            # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            "Authorization": "EMPTY"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.output_len = request_func_input.output_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                              "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def benchmark(
    input_requests: List[RequestFuncInput],
    time_interval: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, time_interval):
        task = asyncio.create_task(async_request_openai_completions(request))
        tasks.append(task)
    outputs = await asyncio.gather(*tasks)
    return outputs


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            try:
                assert output_len == outputs[i].output_len
            except:
                pass
                # We use ignore_eos = True, so we can assert actual output len is what we want.
                # if assert false, mostly because the tokenizer and detokenizer process.

            actual_output_lens.append(outputs[i].output_len)
            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.median(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.mean(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def main(args):
    api_url = f"http://{args.host}:{args.port}/v1/completions"

    input_tokens = args.input_tokens
    output_tokens = args.output_tokens

    input_list, output_list = sample_requests(
        args.num_prompts,
        input_tokens,
        output_tokens,
        args.max_input_tokens,
        args.max_output_tokens,
        args.input_output_type,
        args.seed,
    )

    prompts = []
    for i in range(args.num_prompts):
        if input_list[i][0] is None:
            prompts.append(
                RequestFuncInput(
                    np.random.randint(6023, 6024, input_list[i][1]).tolist(),
                    api_url,
                    input_list[i][1],
                    output_list[i],
                    args.model,
                    1,
                    None,
                    None,
                    True,
                )
            )
        else:
            # this pass is used for test str input.
            prompts.append(
                RequestFuncInput(
                    input_list[i][0],
                    api_url,
                    input_list[i][1],
                    output_list[i],
                    args.model,
                    1,
                    None,
                    None,
                    True,
                )
            )
    # warm up
    await benchmark(prompts[:1], args.time_interval)

    # benchmark
    benchmark_start_time = time.perf_counter()
    outputs = await benchmark(prompts, args.time_interval)
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    selected_percentile_metrics = args.percentile_metrics.split(",")
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))
    
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)
    
    if args.target is not None:
        target_qps = args.target
        if metrics.output_throughput < target_qps:
            print(
                "actual qps: {} < target qps: {} , fail!!".format(
                    metrics.output_throughput, target_qps
                )
            )
            exit(1)
        else:
            print(
                "actual qps: {} > target qps: {} , pass!!".format(
                    metrics.output_throughput, target_qps
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=-1,
        help="Use for generate random length of input, limit min input length",
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=-1,
        help="Use for generate random length of output, limit max output length",
    )
    parser.add_argument("--input-output-type", type=str, default="fix",choices=['fix','normal','uniform'])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--time-interval", type=float, default=0.0)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))

# 测试num-prompts 1 input-tokens 256 --output-tokens 256 作为最基础的基准
# 测试num-prompts 16 input-tokens 2048 --output-tokens 1024 作为一般性能测试
# 测试num-prompts 100 input-tokens 512 max-input-tokens 16384 output-tokens 512 max-output-tokens 16384 input-output-type uniform 监测算子对于不同长度的性能