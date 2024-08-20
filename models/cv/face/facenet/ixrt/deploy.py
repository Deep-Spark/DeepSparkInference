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

import onnx
import os
import simplejson as json
import argparse
from onnxsim import simplify
import numpy as np
import shutil
from onnx import numpy_helper
from onnx import  AttributeProto, TensorProto, GraphProto
from load_ixrt_plugin import load_ixrt_plugin
load_ixrt_plugin()

def onnx_sim(onnx_name, save_name):
    #  simplify onnx
    cmd = "onnxsim {} {}".format(onnx_name, save_name)
    os.system(cmd)
    print("[info] onnxsim done!")


def cut_model(onnx_name):
    input_names = ["input"]
    output_names = ["/last_bn/BatchNormalization_output_0"]
    onnx.utils.extract_model(onnx_name, onnx_name, input_names, output_names) 

def fuse_matmul(onnx_name, save_onnx_name):
    find_matmul = 0

    onnx_model = onnx.load(onnx_name)

    graph = onnx_model.graph
    nodes = graph.node

    conv_weights = None
    conv_bias = None
    bn_weights = None
    bn_bias = None
    conv_weights_new = None
    conv_bias_new = None

    pre_node = None
    for i, node in enumerate(nodes):
        if (node.op_type == "Conv"):
            pass
        if (node.op_type == "MatMul"):
            for k, ten in enumerate(graph.initializer):
                if ten.name == node.input[1]:
                    H , W = ten.dims
                    weights = np.fromstring(ten.raw_data, dtype=np.float32)
                    weights = weights.reshape(ten.dims)
                    conv_weights = weights.transpose()
        if (node.op_type == "BatchNormalization" and pre_node.op_type == "MatMul"):
            find_matmul=1            
            weights = None
            bias = None
            mean = None
            var = None

            for k, ten in enumerate(graph.initializer):
                if ten.name == node.input[1]:
                    weights = np.fromstring(ten.raw_data, dtype=np.float32)
                if ten.name == node.input[2]:
                    bias = np.fromstring(ten.raw_data, dtype=np.float32)
                if ten.name == node.input[3]:
                    mean = np.fromstring(ten.raw_data, dtype=np.float32)
                if ten.name == node.input[4]:
                    var = np.fromstring(ten.raw_data, dtype=np.float32)

            bn_weights = np.diag(weights / np.sqrt(var + 1e-8))
            bn_bias = bias - weights * mean / np.sqrt(var + 1e-8)

            conv_weights_new = np.matmul(bn_weights, conv_weights)
            a, b = conv_weights_new.shape
            conv_weights_new = conv_weights_new.reshape((a,b,1,1))
            # conv_bias_new = bn_weights * conv_bias + bn_bias
            conv_bias_new = 0 + bn_bias
            conv_weights_new_initializer = onnx.numpy_helper.from_array(conv_weights_new, name='conv_weights_new')
            graph.initializer.append(conv_weights_new_initializer)
            conv_bias_new_initializer = onnx.numpy_helper.from_array(conv_bias_new, name='conv_bias_new')
            graph.initializer.append(conv_bias_new_initializer)

            pre_node.op_type = "Conv"
            pre_node.input[0] = "/avgpool_1a/GlobalAveragePool_output_0"
            pre_node.input[1] = "conv_weights_new"
            pre_node.input.append("conv_bias_new")
            pre_node.output[0] = "/last_bn/BatchNormalization_output_0"
            dilations = onnx.helper.make_attribute("dilations", [1,1])
            group = onnx.helper.make_attribute("group", 1)
            kernel_shape = onnx.helper.make_attribute("kernel_shape", [1,1])
            pads = onnx.helper.make_attribute("pads", [0,0,0,0])
            strides = onnx.helper.make_attribute("strides", [1,1])

            pre_node.attribute.append(dilations)
            pre_node.attribute.append(group)
            pre_node.attribute.append(kernel_shape)
            pre_node.attribute.append(pads)
            pre_node.attribute.append(strides)
            graph.node.remove(node)

        pre_node = node

    for i, node in enumerate(nodes):
        if (node.name == "Reshape_353"):
            # print("[reshape] : ", node.name)
            graph.node.remove(node)

    if find_matmul==1:
        output = onnx.helper.make_tensor_value_info('/last_bn/BatchNormalization_output_0', TensorProto.FLOAT, [64, 512, 1, 1])
        graph = onnx.helper.make_graph(
            graph.node,
            "facenet model",
            graph.input,
            [output],
            graph.initializer
        )

        info_model = onnx.helper.make_model(graph, producer_name="facenet")
        info_model.opset_import[0].version = 11
        onnx_model = onnx.shape_inference.infer_shapes(info_model)
        
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, save_onnx_name)

def fuse_mul(onnx_name, save_onnx_name):
    onnx_model = onnx.load(onnx_name)

    graph = onnx_model.graph
    nodes = graph.node
    pre_node = None

    for i, node in enumerate(nodes):
        if (node.op_type == "Constant"):
            pass
        
        if (node.op_type == "Mul" and pre_node.op_type == "Conv" ):
            for ten in graph.initializer:
                if ten.name == node.input[1]:
                    scale_name = ten.name
                    scale = np.fromstring(ten.raw_data, dtype=np.float32)

            for k, ten in enumerate(graph.initializer):
                # print(ten.name)
                if ten.name == pre_node.input[1]:
                    weights_name = ten.name
                    weights = np.fromstring(ten.raw_data, dtype=np.float32)
                    weights *= scale
                    graph.initializer[k].raw_data = weights.tobytes()
    
                if ten.name == pre_node.input[2]:
                    bias_name = ten.name
                    bias = np.fromstring(ten.raw_data, dtype=np.float32)
                    # print("bias len: ",len(da))
                    bias *= scale
                    graph.initializer[k].raw_data = bias.tobytes()

            new_conv = pre_node
            new_conv.output[0] = node.output[0]
            graph.node.remove(node)
        pre_node = node

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, save_onnx_name)

def create_graph_json(onnx_name):
    #  create graph json and weights
    graph_path = onnx_name[0:-5] + "_graph.json"
    weight_path = onnx_name[0:-5] + ".weights"
    
    model = onnx.load(onnx_name)
    graph = model.graph
    nodes = graph.node
    initializer = graph.initializer
    value_info = graph.value_info  # Infer shape info

    model_inputs = [tensor.name for tensor in graph.input]
    model_outputs = [tensor.name for tensor in graph.output]

    model = {}
    model["nodes"] = {}
    model["tensors"] = {}
    model["edges"] = {}
    model["output"] = {}
    data_type_table = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        9: "bool",
        10: "float16",
        11: "double",
        12: "uint32",
        13: "uint64",
    }
    input_cache = []
    for item in graph.input:
        if item.type.tensor_type.elem_type in data_type_table:
            cache = {
                "name": item.name,
                "type": data_type_table[item.type.tensor_type.elem_type],
            }
        else:
            cache = {"name": item.name}
        input_cache.append(cache)
    model["input"] = input_cache

    output_cache = []
    for item in graph.output:
        if item.type.tensor_type.elem_type in data_type_table:
            cache = {
                "name": item.name,
                "type": data_type_table[item.type.tensor_type.elem_type],
            }
        else:
            cache = {"name": item.name}
        output_cache.append(cache)
    model["output"] = output_cache

    # find cast dict
    input_cast_dict = {}
    output_cast_dict = {}
    for i, item in enumerate(nodes):
        node_name = item.name
        input_edge_list = list(item.input)
        output_edge_list = list(item.output)
        # remove input and output cast op
        if item.op_type == "Cast":
            if input_edge_list[0] in model_inputs:
                input_cast_dict[output_edge_list[0]] = input_edge_list[0]
            if output_edge_list[0] in model_outputs:
                output_cast_dict[input_edge_list[0]] = output_edge_list[0]

    for i, item in enumerate(nodes):
        node_name = item.name
        input_edge_list = list(item.input)
        output_edge_list = list(item.output)
        # remove input and output cast op
        if item.op_type == "Cast":
            if input_edge_list[0] in model_inputs:
                continue
            if output_edge_list[0] in model_outputs:
                continue

        for idx, edge_name in enumerate(input_edge_list):
            if edge_name in input_cast_dict.keys():
                input_edge_list[idx] = input_cast_dict[edge_name]

        for idx, edge_name in enumerate(output_edge_list):
            if edge_name in output_cast_dict.keys():
                output_edge_list[idx] = output_cast_dict[edge_name]

        # remove mask in EmbedLayerNormalization
        if item.op_type == "EmbedLayerNormalization":
            no_attention_mask_in_Embed = True
            for input_edge in input_edge_list:
                if "attention_mask" in input_edge:
                    input_edge_list.remove(input_edge)
                    no_attention_mask_in_Embed = False
            if no_attention_mask_in_Embed:
                for tensor_name in model_inputs:
                    if "attention_mask" in tensor_name:
                        output_edge_list[1] = tensor_name

        node_dict = {"inputs": input_edge_list, "outputs": output_edge_list}
        node_dict["op_type"] = item.op_type
        attribute_dict = {}

        if node_name == "":
            for input_edge in input_edge_list:
                node_name += input_edge + "_"
            node_name += "to"
            for output_edge in output_edge_list:
                node_name += "_" + output_edge

        for attr in item.attribute:

            if attr.type == onnx.AttributeProto().AttributeType.FLOAT:
                attribute_dict[attr.name] = attr.f
            if attr.type == onnx.AttributeProto().AttributeType.FLOATS:
                attribute_dict[attr.name] = [x for x in attr.floats]
            if attr.type == onnx.AttributeProto().AttributeType.INT:
                attribute_dict[attr.name] = attr.i
            if attr.type == onnx.AttributeProto().AttributeType.INTS:
                attribute_dict[attr.name] = [x for x in attr.ints]
            if attr.type == onnx.AttributeProto().AttributeType.STRING:
                attribute_dict[attr.name] = str(attr.s.decode("UTF-8"))
            if attr.type == onnx.AttributeProto().AttributeType.STRINGS:
                attribute_dict[attr.name] = [str(x.decode("UTF-8")) for x in attr.strings]

        node_dict["attrbiute"] = attribute_dict
        model["nodes"][node_name] = node_dict

    for i, item in enumerate(initializer):
        tensor_name = item.name
        tensor_dict = {}
        if item.data_type in data_type_table:
            tensor_dict["data_type"] = data_type_table[item.data_type]
        else:
            print(
                tensor_name,
                " use unsupport data type: ",
                item.data_type,
                ", data info will not be saved",
            )
            continue
        tensor_dict["dims"] = list(item.dims)

        model["tensors"][tensor_name] = tensor_dict

    with open(graph_path, "w") as fh:
        json.dump(model, fh, indent=4)


    """
    Export weight
    """
    byte_string = "".encode()

    weight_file_postfix = ".weights"
    for item in initializer:
        tensor_name = item.name

        np_data = None
        if len(item.raw_data):
            np_data = np.frombuffer(item.raw_data, dtype=np.byte)
        elif item.data_type == 1 and len(item.float_data):
            np_data = np.array(list(item.float_data), dtype=np.float32)
        elif item.data_type == 2 and len(item.int32_data):
            np_data = np.array(list(item.int32_data), dtype=np.uint8)
        elif item.data_type == 6 and len(item.int32_data):
            np_data = np.array(list(item.int32_data), dtype=np.int32)
        elif item.data_type == 7 and len(item.int64_data):
            np_data = np.array(list(item.int64_data), dtype=np.int64)
        elif item.data_type == 10 and len(item.int32_data):
            np_data = (
                np.asarray(item.int32_data, dtype=np.uint16)
                .reshape(item.dims)
                .view(np.float16)
            )
        else:
            print(
                "tensor name: ",
                tensor_name,
                ", type: ",
                item.data_type,
                ", len: ",
                len(item.raw_data),
                len(item.float_data),
                len(item.int32_data),
                len(item.int64_data),
                ", will not save into weights file",
            )

        if np_data is not None:
            byte_string += np.uint64(len(tensor_name)).tobytes()
            byte_string += tensor_name.encode()
            np_bytes = np_data.tobytes()
            byte_string += np.uint64(len(np_bytes)).tobytes()
            byte_string += np_bytes


    # Export weight values as bin file
    with open(weight_path, "wb") as fh:
        fh.write(byte_string)
    print("----------------------------")
    print("[OK] graph and weights file save at :")
    print(graph_path)
    print(weight_path)
    return graph_path, weight_path

def add_facenet_norm(cfg_name):
    graph_json = json.load(open(cfg_name))

    graph_json["nodes"]["facenet_norm_1"] = {
            "inputs": [
                "/last_bn/BatchNormalization_output_0"
            ],
            "outputs": [
                "/Pow_1_output_0"
            ],
            "op_type": "FacenetNorm",
            "attrbiute": {
                "size": 512
            }
        }
    graph_json["output"] = []
    graph_json["output"].append({"name":"/Pow_1_output_0", "type":"float32"})

    with open(cfg_name, "w") as fh:
        json.dump(graph_json, fh, indent=4)


def main(args):
    print("[info] input onnx name :", args.onnx_name)
    # onnxsim
    onnx_sim(args.onnx_name, "tmp1.onnx")
    # cut model
    cut_model("tmp1.onnx")
    # fuse matmul bn
    fuse_matmul("tmp1.onnx", "tmp2.onnx")
    # fuse mul
    fuse_mul("tmp2.onnx", "facenet_weights/facenet.onnx")
    # generate cfg weights
    # graph_path, weight_path = create_graph_json("facenet_weights/facenet.onnx")
    # add facenet norm
    # add_facenet_norm(graph_path)

    os.remove("tmp1.onnx")
    os.remove("tmp2.onnx")
    print("\n[info] facenet deploy done!!!")


def parse_args():
    parser = argparse.ArgumentParser("deploy facenet")
    parser.add_argument("--model_name", default="facenet", help="model name")
    parser.add_argument("--onnx_name", default="facenet_weights/facenet_export.onnx", help="onnx filepath")
    parser.add_argument("--save_name", default="facenet_weights/facenet.onnx", help="onnx filepath")
    parser.add_argument("--data_type", default="int8", type=str, choices=["float16", "int8"], help="int8 float16")
    parser.add_argument("--batch_size", default="64", type=int, help="batch_size")
    parser.add_argument("--quant_file", default="", type=str, help="quant file")
    parser.add_argument("--img_size", default="160", type=int, help="image size")
    parser.add_argument("--device", default=0, type=int, help="cuda device 0 1 3 ...")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)