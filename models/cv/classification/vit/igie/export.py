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
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="export onnx model path.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    checkpoint = "vit-base-patch16-224"

    # 1. 加载模型和预处理器
    print(f"Loading model: {checkpoint}")
    feature_extractor = ViTImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(checkpoint)

    # 2. 设置模型为评估模式 
    model.eval()

    save_path = Path(args.output)

    # 3. 构造一个标准的 Dummy Input
    # ViT 标准输入格式: (Batch_Size, Channels, Height, Width)
    # 这里固定为 1, 3, 224, 224
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Starting PyTorch Native Export...")

    # 4. 使用 PyTorch 原生导出
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,        # 导出训练参数（权重）
            opset_version=17,          # 使用较新的 opset
            do_constant_folding=True,  # 优化常数折叠
            input_names=['pixel_values'],  # <--- 关键修复：必须与推理脚本预期的名称一致
            output_names=['output'],   # 指定输出名称
            dynamic_axes={             # 定义动态轴
                'pixel_values': {0: 'batch_size'}, # <--- 关键修复：键名必须与 input_names 对应
                'output': {0: 'batch_size'}
            },
            dynamo=False               # 强制使用旧版稳定导出器，避开 PyTorch 2.7 的 bug
        )
        print(f"✅ Export onnx model successfully to {save_path}!")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("Tip: If error persists, try removing 'dynamo=False' or downgrading PyTorch.")

if __name__ == "__main__":
    main()
