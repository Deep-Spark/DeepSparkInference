#!/bin/bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
#All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # 遇到错误立即退出

# 获取 commit-msg 阶段传入的提交信息文件路径
COMMIT_MSG_FILE="${1:-}"

# 提取本次提交的作者邮箱（优先从提交上下文取，兜底取本地配置）
AUTHOR_EMAIL=$(git log -1 --pretty=format:"%ae" HEAD@{0} 2>/dev/null)
if [ -z "$AUTHOR_EMAIL" ]; then
    AUTHOR_EMAIL=$(git config --get user.email)
fi

# 校验邮箱非空
if [ -z "$AUTHOR_EMAIL" ]; then
    echo -e "\033[31m❌ 错误：未配置 Git 提交邮箱！\033[0m"
    echo "请执行：git config --local user.email '你的合规邮箱@xxx.com'"
    exit 1
fi

# 定义允许的邮箱正则（按需修改）
ALLOWED_PATTERN="^.*@iluvatar\.com$"

# 校验邮箱格式
if [[ ! "$AUTHOR_EMAIL" =~ $ALLOWED_PATTERN ]]; then
    echo -e "\033[31m❌ 提交邮箱不符合规范！\033[0m"
    echo "当前邮箱：$AUTHOR_EMAIL"
    echo "允许的邮箱：仅支持 @iluvatar.com 邮箱"
    exit 1
fi

echo -e "\033[32m✅ 提交邮箱校验通过：$AUTHOR_EMAIL\033[0m"
exit 0
