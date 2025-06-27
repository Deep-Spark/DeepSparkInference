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

import csv
import json
import re


TRANSL = {
    "LLM (Large Language Model)": "大语言模型（LLM）",
    "Computer Vision": "计算机视觉（CV）",
    "Classification": "视觉分类",
    "Object Detection": "目标检测",
    "Face Recognition": "人脸识别",
    "OCR (Optical Character Recognition)": "光学字符识别（OCR）",
    "Pose Estimation": "姿态估计",
    "Instance Segmentation": "实例分割",
    "Semantic Segmentation": "语义分割",
    "Multi-Object Tracking": "多目标跟踪",
    "Multimodal": "多模态",
    "NLP": "自然语言处理（NLP）",
    "PLM (Pre-trained Language Model)": "预训练语言模型（PLM）",
    "Audio": "语音",
    "Speech Recognition": "语音识别",
    "Others": "其他",
    "Recommendation Systems": "推荐系统",
}

CATEG_TRANSL = {
    "cv": "Computer Vision",
    "nlp": "NLP",
    "others": "Others",
    "multimodal": "Multimodal",
    "audio": "Audio",
    "llm": "LLM (Large Language Model)",
    "classification": "Classification",
    "object_detection": "Object Detection",
    "ocr": "OCR (Optical Character Recognition)",
    "instance_segmentation": "Instance Segmentation",
    "pose_estimation": "Pose Estimation",
    "plm": "PLM (Pre-trained Language Model)",
    "recommendation": "Recommendation Systems",
    "vision_language_model": "Vision Language Model",
    "speech_recognition": "Speech Recognition",
    "face_recognition": "Face Recognition",
    "multi_object_tracking": "Multi-Object Tracking",
    "diffusion_model": "Diffusion Model",
    "semantic_segmentation": "Semantic Segmentation",
}

FW_DISPLAY = {
    "vllm": "vLLM",
    "igie": "IGIE",
    "ixrt": "IxRT",
    "trtllm": "TRT-LLM",
    "tgi": "TGI",
    "ixformer": "IxFormer",
    "diffusers": "Diffusers",
}

HEADER_DISPLAY = {
    "display_name": "模型名称",
    "framework": "框架",
    "release_version": "发布版本",
    "release_sdk": "发布SDK",
    "release_gpgpu": "发布GPGPU",
    "latest_sdk": "最新SDK",
    "latest_gpgpu": "最新GPGPU",
    "category": "分类",
    "toolbox": "工具箱",
    "mdims": "多维度评测",
    "dataset": "数据集",
    # "license": "许可证",
    "model_path": "模型路径",
    "readme_file": "README链接",
    # "bitbucket_repo": "Bitbucket仓库",
    # "bitbucket_branch": "Bitbucket分支",
    # "bitbucket_path": "Bitbucket路径",
    # "develop_owner": "开发负责人",
    # "github_repo": "GitHub仓库",
    # "github_branch": "GitHub分支",
    # "github_path": "GitHub路径",
}


# Define value transformation function
def transform_value(key, value):
    if key == "category":
        category = "/".join([CATEG_TRANSL[c] for c in value.split("/")])
        return re.sub(r" \(.+\)", "", category)
    elif key == "precisions":
        return ", ".join(value).upper()  # 列表元素处理
    elif key == "framework":
        return FW_DISPLAY[value.lower()]
    # elif isinstance(value, str):
    #     return f"PREFIX_{value}"  # 字符串加前缀
    return value  # 其他情况保持原样


with open("tests/model_info.json") as f:
    models = json.load(f)["models"]

# Extract CSV headers (automatically get all dictionary keys)
headers = [h for h in list(models[0].keys()) if h in HEADER_DISPLAY]

processed = [
    {k: transform_value(k, v) for k, v in m.items() if k in HEADER_DISPLAY}
    for m in models
]

# Write to CSV file
csv_file = "tests/models.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(processed)

with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    rows = list(reader)

# Check Header length
new_headers = list(HEADER_DISPLAY.values())
if len(new_headers) != len(rows[0]):
    raise ValueError("新表头的长度必须与原表头一致")

# change to chinese header
rows[0] = new_headers

# 写回文件
with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
