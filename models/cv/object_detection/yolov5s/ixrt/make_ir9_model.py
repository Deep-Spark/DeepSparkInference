# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

# Fix onnxruntime 1.17.1: Unsupported model IR version: 10, max supported IR version: 9
import argparse
from pathlib import Path

def make_ir9_model(input_path, output_path=None):
    """Downgrade an ONNX model's IR version to 9 for onnxruntime <= 1.17.1.

    Args:
        input_path: Path to the input ONNX model file.
        output_path: Optional path to save the IR9 model. If ``None`` (default),
            a sibling path ``<input_stem>_ir9<input_suffix>`` is used and reused
            when it already exists.

    Returns:
        Path to the IR9-compatible ONNX model as a string. Returns
        ``input_path`` unchanged when the model is already at IR version 9
        or below.
    """
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {src}")

    if output_path is not None:
        dst = Path(output_path)
        if dst == src:
            raise ValueError(f"output_path must differ from input_path: {src}")
    else:
        dst = src.with_name(f"{src.stem}_ir9{src.suffix}")
        if dst.exists():
            return str(dst)

    import onnx

    model = onnx.load(str(src))
    if model.ir_version <= 9:
        return str(src)

    model.ir_version = 9
    onnx.save(model, str(dst))
    print(f"saved ONNX IR9 compatibility copy: {dst}")
    return str(dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to IR version 9 for onnxruntime <= 1.17.1.",
    )
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        help="Path to the input ONNX model.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        help="Path to save the IR9 model. Defaults to <input_stem>_ir9<input_suffix>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    make_ir9_model(args.input_path, args.output_path)


if __name__ == "__main__":
    main()