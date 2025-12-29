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
import onnx
from onnx import helper

def modify_onnx(args):
    # 加载现有的ONNX模型
    model = onnx.load(args.src)
    # 创建一个新的节点
    new_node_0 = helper.make_node(
        'YoloXDecoder',  # 你的新操作类型  
        inputs=['/head/reg_preds.0/Conv_output_0', '/head/obj_preds.0/Conv_output_0', '/head/cls_preds.0/Conv_output_0'],  # 原始输出的名称  
        outputs=['decoder_8'],  # 新节点的输出名称   
        faster_impl=1,
        num_class=80,
        stride=8,
        name='ixq_YoloXDecoder_0',  # 新节点的名称   
    )  
    # 将新节点添加到模型的graph中
    model.graph.node.append(new_node_0)

    new_node_1 = helper.make_node(
        'YoloXDecoder',  # 你的新操作类型  
        inputs=['/head/reg_preds.1/Conv_output_0', '/head/obj_preds.1/Conv_output_0', '/head/cls_preds.1/Conv_output_0'],  # 原始输出的名称  
        outputs=['decoder_16'],  # 新节点的输出名称  
        faster_impl=1,
        num_class=80,
        stride=16,
        name='ixq_YoloXDecoder_1'  # 新节点的名称    
    ) 
    # 将新节点添加到模型的graph中
    model.graph.node.append(new_node_1) 

    new_node_2 = helper.make_node(
        'YoloXDecoder',  # 你的新操作类型  
        inputs=['/head/reg_preds.2/Conv_output_0', '/head/obj_preds.2/Conv_output_0', '/head/cls_preds.2/Conv_output_0'],  # 原始输出的名称  
        outputs=['decoder_32'],  # 新节点的输出名称
        faster_impl=1,
        num_class=80,
        stride=32,
        name='ixq_YoloXDecoder_2'  # 新节点的名称    
    ) 

    # 将新节点添加到模型的graph中
    model.graph.node.append(new_node_2) 

    new_node_3 = helper.make_node(
        'Concat',  # 你的新操作类型  
        inputs=['decoder_8', 'decoder_16', 'decoder_32'],  # 原始输出的名称  
        outputs=['output_new'],  # 新节点的输出名称  
        name='ixq_Concat_0'  # 新节点的名称  
    ) 
    # 将新节点添加到模型的graph中
    model.graph.node.append(new_node_3) 

    # model.graph.output.clear()

    # 要设置为输出的节点的名称  
    node_output_name = 'output_new'  
    
    # 检查是否已存在该输出的ValueInfoProto  
    output_value_info = None  
    for output in model.graph.output:  
        if output.name == node_output_name:  
            output_value_info = output  
            break  
    
    # 如果不存在，则创建ValueInfoProto  
    if output_value_info is None:  
        # 假设我们知道输出的类型（这里以float为例）  
        tensor_type = onnx.TensorProto.FLOAT  
        # 创建ValueInfoProto  
        output_value_info = helper.make_tensor_value_info(  
            name=node_output_name,  
            elem_type=tensor_type,  
            shape=None  # 或者你可以指定具体的shape  
        )  
    
    # 将ValueInfoProto添加到model.graph.output列表中  
    model.graph.output.append(output_value_info)  

    output = model.graph.output

    for i in range(len(output)-1):
        model.graph.output.remove(output[0])
    # print(output[9])

    # 保存修改后的模型
    onnx.save(model, args.dst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    modify_onnx(args)
    print("Surged onnx lies on", args.dst)