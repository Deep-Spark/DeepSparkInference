#!/bin/bash
# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
#

"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for multimodal embedding.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from argparse import Namespace
from typing import Literal, NamedTuple, Optional, TypedDict, Union, get_args
import io
import base64
from PIL import Image
from vllm import LLM
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
from vllm import LLM, EngineArgs
import dataclasses

class TextQuery(TypedDict):
    modality: Literal["text"]
    text: str


class ImageQuery(TypedDict):
    modality: Literal["image"]
    image: Image.Image


class TextImageQuery(TypedDict):
    modality: Literal["text+image"]
    text: str
    image: Image.Image


QueryModality = Literal["text", "image", "text+image"]
Query = Union[TextQuery, ImageQuery, TextImageQuery]


class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    image: Optional[Image.Image]


def run_e5_v(query: Query, engine_params):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'  # noqa: E501

    if query["modality"] == "text":
        text = query["text"]
        prompt = llama3_template.format(
            f"{text}\nSummary above sentence in one word: ")
        image = None
    elif query["modality"] == "image":
        prompt = llama3_template.format(
            "<image>\nSummary above image in one word: ")
        image = query["image"]
    else:
        modality = query['modality']
        raise ValueError(f"Unsupported query modality: '{modality}'")

    llm = LLM(**engine_params)

    return ModelRequestData(
        llm=llm,
        prompt=prompt,
        image=image,
    )



def get_query(modality: QueryModality):
    if modality == "text":
        return TextQuery(modality="text", text="A dog sitting in the grass")


    if modality == "image":
        image: Image = Image.open("vllm_public_assets/American_Eskimo_Dog.jpg")
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return ImageQuery(
            modality="image",
            image= fetch_image(f"data:image/jpeg;base64,{image_base64}"
            ),
        )

    if modality == "text+image":
        image: Image = Image.open("vllm_public_assets/Felis_catus-cat_on_snow.jpg")
        image = image.convert("RGB")
        image_data = io.BytesIO()
        image.save(image_data, format='JPEG')
        image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return TextImageQuery(
            modality="text+image",
            text="A cat standing in the snow.",
            image= fetch_image(f"data:image/jpeg;base64,{image_base64}"
            ),
        )

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def run_encode(engine_params, modality: QueryModality):
    query = get_query(modality)
    req_data = run_e5_v(query, engine_params)

    mm_data = {}
    if req_data.image is not None:
        mm_data["image"] = req_data.image

    outputs = req_data.llm.embed({
        "prompt": req_data.prompt,
        "multi_modal_data": mm_data,
    })

    for output in outputs:
        print(output.outputs.embedding)
        if output.outputs.embedding is not None:
            print("Offline inference is successful!")



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for multimodal embedding')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=get_args(QueryModality),
                        help='Modality of the input.')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = [attr.name for attr in dataclasses.fields(EngineArgs)]
    engine_params = {attr: getattr(args, attr) for attr in engine_args}
    
    run_encode(engine_params, args.modality)
