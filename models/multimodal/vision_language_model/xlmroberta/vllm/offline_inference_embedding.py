# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
import time
from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model="intfloat/e5-small",
        runner="pooling",
        enforce_eager=True,
    )
    return parser.parse_args()


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass runner="pooling" for embedding models
    llm = LLM(**vars(args))

    num_tokens = 0
    start_time = time.perf_counter()
    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = llm.encode(prompts)
    end_time = time.perf_counter()
    duration_time = end_time - start_time

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        num_tokens += len(output.outputs.embedding)
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)
    num_requests = len(prompts)  # 请求的数量
    qps = num_requests / duration_time
    print(f"requests: {num_requests}, QPS: {qps}, tokens: {num_tokens}, Token/s: {num_tokens/duration_time}")


if __name__ == "__main__":
    args = parse_args()
    main(args)