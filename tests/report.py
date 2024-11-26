# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import json

def generate_report(all_results):
    print(all_results)
    # 区分两种数据结构的键
    detection_keys = {"IoU=0.50:0.95", "IoU=0.50"}
    classification_keys = {"Top1 acc", "Top5 acc"}

    # 生成 HTML 表格
    html_output = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>JSON to HTML Table</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
    """
    html_detec_body = """
        <h1>Detection Results</h1>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Precision</th>
                    <th>Status</th>
                    <th>Mean inference time</th>
                    <th>Mean fps</th>
                    <th>Cost time (s)</th>
                    <th>IoU=0.50:0.95</th>
                    <th>IoU=0.50</th>
                </tr>
            </thead>
            <tbody>
    """
    html_tbody = """
            </tbody>
        </table>
    """
    html_clf_body = """
        <h1>Classification Results</h1>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Precision</th>
                    <th>Status</th>
                    <th>Mean inference time</th>
                    <th>Mean fps</th>
                    <th>Cost time (s)</th>
                    <th>Top1 acc</th>
                    <th>Top5 acc</th>
                </tr>
            </thead>
            <tbody>
    """

    html_detec_tr_content = ""
    html_clf_tr_content = ""
    # 填充检测结果的表格
    for item in all_results:
        for precision, result in item['result'].items():
            td_status = f"""<td style="color:blue">{result['status']}</td>"""
            if result['status'] != "PASS":
                td_status = f"""<td style="color:red">{result['status']}</td>"""

            row = f"""
                <tr>
                    <td>{item['name']}</td>
                    <td>{precision}</td>
                    {td_status}
                    <td>{result['Mean inference time']}</td>
                    <td>{result['Mean fps']}</td>
                    <td>{result['Cost time (s)']}</td>
            """
            if any(key in result for key in detection_keys):
                row += f"""        <td>{result['IoU=0.50:0.95']}</td>
                <td>{result['IoU=0.50']}</td>
            </tr>
                """
                html_detec_tr_content += row
            
            if any(key in result for key in classification_keys):
                row += f"""        <td>{result['Top1 acc']}</td>
                <td>{result['Top5 acc']}</td>
            </tr>
                """
                html_clf_tr_content += row

    if html_detec_tr_content != "":
        html_output += html_detec_body + html_detec_tr_content + html_tbody
        
    if html_clf_tr_content != "":
        html_output += html_clf_body + html_clf_tr_content + html_tbody

    html_output += """
    </body>
    </html>
    """

    # 将HTML内容写入文件
    html_file_path = '/mnt/deepspark/ci_report/output.html'
    with open(html_file_path, 'w') as f:
        f.write(html_output)

    print(f"HTML file has been written to {html_file_path}")
