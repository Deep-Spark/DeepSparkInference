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
from pathlib import Path
from transformers.onnx import export
from transformers.models.bert import BertOnnxConfig

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    checkpoint = "neuralmagic/bert-large-uncased-finetuned-squadv1"
    feature_extractor = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    save_path = Path(args.output)
    onnx_config = BertOnnxConfig(model.config)

    # export onnx model
    export(
        feature_extractor, model, onnx_config,
        onnx_config.default_onnx_opset, save_path
    )
    
    print("Export onnx model successfully! ")

if __name__ == "__main__":
    main()