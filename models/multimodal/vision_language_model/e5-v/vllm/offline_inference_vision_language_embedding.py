# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal pooling.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import io
import base64
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, TypedDict, get_args

from PIL.Image import Image, open as ImageOpen

from vllm import LLM, EngineArgs
from vllm.entrypoints.score_utils import ScoreMultiModalParam
from vllm.multimodal.utils import fetch_image
from vllm.utils.argparse_utils import FlexibleArgumentParser

ROOT_DIR = Path(__file__).parent.parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"


class TextQuery(TypedDict):
    modality: Literal["text"]
    text: str


class ImageQuery(TypedDict):
    modality: Literal["image"]
    image: Image


class TextImageQuery(TypedDict):
    modality: Literal["text+image"]
    text: str
    image: Image


class TextImagesQuery(TypedDict):
    modality: Literal["text+images"]
    text: str
    image: ScoreMultiModalParam


QueryModality = Literal["text", "image", "text+image", "text+images"]
Query: TypeAlias = TextQuery | ImageQuery | TextImageQuery | TextImagesQuery


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    image: Image | None = None
    query: str | None = None
    documents: ScoreMultiModalParam | None = None

def run_e5_v(query: Query) -> ModelRequestData:
    llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"  # noqa: E501

    if query["modality"] == "text":
        text = query["text"]
        prompt = llama3_template.format(f"{text}\nSummary above sentence in one word: ")
        image = None
    elif query["modality"] == "image":
        prompt = llama3_template.format("<image>\nSummary above image in one word: ")
        image = query["image"]
    else:
        modality = query["modality"]
        raise ValueError(f"Unsupported query modality: '{modality}'")

    engine_args = EngineArgs(
        model="royokong/e5-v",
        runner="pooling",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image=image,
    )

def get_query(modality: QueryModality):
    if modality == "text":
        return TextQuery(modality="text", text="A dog sitting in the grass")

    if modality == "image":
        image: Image = ImageOpen("vllm_public_assets/American_Eskimo_Dog.jpg")
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return ImageQuery(
            modality="image",
            image= fetch_image(f"data:image/jpeg;base64,{image_base64}"
            ),
        )

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(model: str, modality: QueryModality, seed: int | None):
    query = get_query(modality)
    req_data = model_example_map[model](query)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": seed}
    llm = LLM(**engine_args)

    mm_data = {}
    if req_data.image is not None:
        mm_data["image"] = req_data.image

    outputs = llm.embed(
        {
            "prompt": req_data.prompt,
            "multi_modal_data": mm_data,
        }
    )

    print("-" * 50)
    for output in outputs:
        print(output.outputs.embedding)
        print("-" * 50)
        if output.outputs.embedding is not None:
            print("Offline inference is successful!")

model_example_map = {
    "e5_v": run_e5_v
}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for multimodal pooling tasks."
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="vlm2vec_phi3v",
        choices=model_example_map.keys(),
        help="The name of the embedding model.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="embedding",
        choices=["embedding", "scoring"],
        help="The task type.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=get_args(QueryModality),
        help="Modality of the input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    return parser.parse_args()


def main(args: Namespace):
    if args.task == "embedding":
        run_encode(args.model_name, args.modality, args.seed)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    args = parse_args()
    main(args)