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
import base64
from PIL import Image
import io

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
    content: dict
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


def sample_requests(
    num_requests: int,
    output_tokens: int,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[int]]:
    np.random.seed(seed)
    random.seed(seed)

    input_len_list = [["What's in this image?", None] for _ in range(num_requests)]
    output_len_list = [output_tokens for _ in range(num_requests)]

    for inputs in input_len_list:
        assert len(inputs) == 2
        # [str, int] or [None ,int]
        assert isinstance(inputs[0], str) or inputs[0] is None
        assert isinstance(inputs[1], int) or inputs[1] is None
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
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.content
                },
            ],
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        headers = {
            "Content-Type": "application/json",
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
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)
                                
                                generated_text += delta["content"]
                            most_recent_timestamp = timestamp

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
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    output_tokens = args.output_tokens
    if args.chat_template is not None:
        tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code, chat_template=args.chat_template)
    else:
        tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    input_list, output_list = sample_requests(
        args.num_prompts,
        output_tokens,
        args.seed,
    )

    prompts = []
    image_size = args.image_size.split(",")
    image_size = [int(x) for x in image_size]
    image = Image.open(args.image_path)
    image = image.resize(image_size)
    image = image.convert("RGB")
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
    for i in range(args.num_prompts):
        messages = [{'role': 'user', 'content': f"{input_list[i][0]}"}]
        content = [
            {
                "type": "text", 
                "text": input_list[i][0]
            }
        ]
        prompt_ids = tokenizer.apply_chat_template(messages,tokenize=True)
        assert isinstance(prompt_ids, list) and isinstance(prompt_ids[0], int)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            }
        )
        
        prompts.append(
            RequestFuncInput(
                content,
                api_url,
                len(prompt_ids),
                output_list[i],
                args.model,
                1,
                None,
                None,
                True,
            )
        )
    # warm up
    outputs = await benchmark(prompts[:16], args.time_interval)

    # benchmark
    benchmark_start_time = time.perf_counter()
    outputs = await benchmark(prompts, args.time_interval)
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

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
    parser.add_argument("--output-tokens", type=int, default=128)
    # this paramater could be unnecessary, because most multimodal's preprocessor will resize image before using it.
    parser.add_argument("--image-path", type=str, default="test.jpg")
    parser.add_argument("--image-size", type=str, default="512,512")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--time-interval", type=float, default=0.0)
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--trust-remote-code', type=bool, default=False)
    parser.add_argument('--chat-template',type=str,default=None)
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
