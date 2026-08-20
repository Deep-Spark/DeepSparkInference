# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
from pathlib import Path
from optimum.exporters.onnx import main_export

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="export onnx model path (directory or .onnx file).")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # 建议使用完整的 hub 模型 ID
    checkpoint = "clip-vit-base-patch32"
    
    save_path = Path(args.output)
    # optimum 的 main_export 默认输出到一个目录
    output_dir = save_path.parent if save_path.suffix == ".onnx" else save_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {checkpoint} to {output_dir}...")
    
    # 执行导出
    main_export(
        model_name_or_path=checkpoint,
        output=str(output_dir),
        task="zero-shot-image-classification",
        opset=14,  # 可根据需要调整 opset 版本
    )
    
    # 如果用户指定了具体的 .onnx 文件名，则将默认生成的 model.onnx 重命名
    if save_path.suffix == ".onnx":
        default_onnx = output_dir / "model.onnx"
        if default_onnx.exists():
            default_onnx.rename(save_path)
            print(f"✅ Model successfully exported to {save_path}")
        else:
            print(f"⚠️ Export completed, but default model.onnx not found in {output_dir}")

if __name__ == "__main__":
    main()