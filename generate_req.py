import os
import re
import yaml

# YAML 文件路径
yaml_file_path = "models_igie.yaml"

# 初始化结果列表
results = []

# 遍历当前目录及其子目录
for root, dirs, files in os.walk("."):
    # 检查子目录是否存在名为 "igie" 的目录
    if 'igie' in dirs:
        igie_path = os.path.join(root, 'igie')
        relative_path = os.path.relpath(igie_path, '.')
        parent_dir_name = os.path.basename(root)
        readme_path = os.path.join(igie_path, 'README.md')

        # 初始化字段
        download_url = None
        datasets = None
        precisions = []

        # 解析 README.md 文件
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as readme_file:
                for line in readme_file:
                    # 查找 "Pretrained model: " 开头的行，提取 <> 中的内容
                    if line.startswith("Pretrained model: "):
                        match = re.search(r'<(.*?)>', line)
                        if match:
                            download_url = match.group(1)
                    
                    # 查找 "Dataset:" 开头的行，提取 <> 中的内容
                    elif line.startswith("Dataset: "):
                        match = re.search(r'<(.*?)>', line)
                        if match:
                            datasets = match.group(1)
                    
                    # 检查是否包含 fp16 或 int8
                    if 'fp16' in line.lower() and 'fp16' not in precisions:
                        precisions.append('fp16')
                    if 'int8' in line.lower() and 'int8' not in precisions:
                        precisions.append('int8')

        # 添加结果到列表
        result = OrderedDict([
            ('name', parent_dir_name),
            ('relative_path', relative_path),
            ('task_type', '/'.join(relative_path.split('/')[1:3]) if len(relative_path.split('/')) > 2 else None),
            ('download_url', download_url),
            ('datasets', datasets),
            ('precisions', precisions)
        ])
        results.append(result)

# 将结果写入 YAML 文件
with open(yaml_file_path, 'w', encoding='utf-8', newline='\n') as yaml_file:
    yaml.dump(results, yaml_file, default_flow_style=False, allow_unicode=True)

print(f"YAML 文件已生成: {yaml_file_path}")
