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

import argparse

import torch
import torch.nn as nn
import onnx

from transformers import DINOv3ViTModel

class DINOv3ONNXWrapper(nn.Module):
    """
    Export full HuggingFace DINOv3 ViT backbone.

    ONNX input:
        images:
            [B, 3, H, W]

    ONNX outputs:
        last_hidden_state:
            [B, tokens, hidden_size]

        pooler_output:
            [B, hidden_size]
    """

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, images):

        outputs = self.model(
            pixel_values=images,
            return_dict=False,
        )

        last_hidden_state = outputs[0]
        pooler_output = outputs[1]

        return (
            last_hidden_state,
            pooler_output,
        )

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local HuggingFace DINOv3 model directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dinov3_vits16.onnx",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()
    
    print()
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")

    model = DINOv3ViTModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        attn_implementation="eager",
    )

    model = model.float()

    model.eval()

    config = model.config

    print("=" * 80)
    print("DINOv3 Config")
    print("=" * 80)

    print(f"model_type          : {config.model_type}")
    print(f"image_size          : {config.image_size}")
    print(f"patch_size          : {config.patch_size}")

    print(f"hidden_size         : {config.hidden_size}")
    print(f"intermediate_size   : {config.intermediate_size}")

    print(f"num_hidden_layers   : {config.num_hidden_layers}")
    print(f"num_attention_heads : {config.num_attention_heads}")

    print(
        f"num_register_tokens : "
        f"{config.num_register_tokens}"
    )

    print(
        f"attention impl      : "
        f"{config._attn_implementation}"
    )

    wrapper = DINOv3ONNXWrapper(
        model
    )

    wrapper.eval()

    images = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        dtype=torch.float32,
    )


    dynamic_shapes = None

    if args.dynamic_batch:

        batch = torch.export.Dim(
            "batch",
            min=1,
        )

        dynamic_shapes = {
            "images": {
                0: batch,
            }
        }


    print()
    print("=" * 80)
    print("Export ONNX")
    print("=" * 80)

    print(f"Output       : {args.output}")
    print(f"Input shape  : {tuple(images.shape)}")
    print(f"Opset        : {args.opset}")
    print(f"Dynamic batch: {args.dynamic_batch}")

    with torch.inference_mode():

        torch.onnx.export(
            wrapper, (images,), args.output,
            input_names=[
                "images",
            ],
            output_names=[
                "last_hidden_state",
                "pooler_output",
            ],
            opset_version=args.opset,
            dynamo=False,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )


    print("=" * 80)

    onnx_model = onnx.load(
        args.output
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("ONNX checker: PASSED")


if __name__ == "__main__":
    main()
